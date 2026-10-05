//! PDH queries: `GPU Engine` (per-process per-engine utilization) and
//! `GPU Process Memory` (per-process dedicated/shared usage, KB).
//!
//! Instance name formats (Windows 10 17063+):
//! - GPU Engine:         `pid_<n>_luid_0x<high32>_0x<low32>_phys_<n>_eng_<n>_engtype_<T>`
//! - GPU Process Memory: `pid_<n>_luid_0x<high32>_0x<low32>_phys_<n>`
//!
//! Engine types seen in practice: 3D, Compute, Copy, VideoDecode, VideoEncode,
//! VideoProcessing (also Crypto/Photo/... mapped to `other` upstream).

use std::collections::HashMap;
use windows::core::{Error as WinError, w};
use windows::Win32::System::Performance::{
    PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
    PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE, PDH_HCOUNTER, PDH_HQUERY,
    PDH_MORE_DATA,
};

pub struct EngineSample {
    pub pid: u32,
    /// (low, high) parts of the adapter LUID
    pub luid: Option<(u32, u32)>,
    pub engtype: String,
    /// % (0-100 per engine instance)
    pub utilization: f64,
}

pub struct ProcMemSample {
    pub pid: u32,
    pub luid: Option<(u32, u32)>,
    /// committed dedicated (VRAM), bytes — includes evictable driver buffers
    /// like ReLive replay (~10 GiB on AMDRSServ but ~0 resident)
    pub dedicated_commit_bytes: u64,
    /// committed shared memory, bytes
    pub shared_commit_bytes: u64,
    /// resident local (VRAM) bytes — "Local Usage", ≈ drm-resident-vram
    pub local_resident_bytes: u64,
    /// resident non-local (system/shared) bytes — "Non Local Usage"
    pub non_local_resident_bytes: u64,
}

/// u32 PDH status code: 0 == ERROR_SUCCESS
fn ok(status: u32) -> bool { status == 0 }

/// Parse `pid_<n>_luid_0x<h>_0x<l>...` tail segments.
fn parse_instance(name: &str) -> (u32, Option<(u32, u32)>) {
    let mut pid = None;
    let mut luid = None;
    let mut it = name.split('_');

    while let Some(seg) = it.next() {
        match seg {
            "pid" => pid = it.next().and_then(|s| s.parse().ok()),
            "luid" => {
                let high = it.next();
                let low = it.next();
                if let (Some(h), Some(l)) = (high, low) {
                    let h = u32::from_str_radix(h.trim_start_matches("0x"), 16).ok();
                    let l = u32::from_str_radix(l.trim_start_matches("0x"), 16).ok();
                    if let (Some(h), Some(l)) = (h, l) {
                        luid = Some((l, h)); // (low, high)
                    }
                }
            },
            _ => {},
        }
    }

    (pid.unwrap_or(0), luid)
}

fn engtype_of(name: &str) -> String {
    if let Some(pos) = name.find("engtype_") {
        name[pos + "engtype_".len()..].to_string()
    } else {
        "Unknown".to_string()
    }
}

/// Read all wildcard instances of a formatted counter array.
fn read_counter_array(counter: PDH_HCOUNTER) -> Result<Vec<(String, i64)>, u32> {
    let mut buffer_size: u32 = 0;
    let mut item_count: u32 = 0;

    // First call: probe size.
    let status = unsafe {
        PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &mut buffer_size, &mut item_count, None)
    };
    if status != PDH_MORE_DATA {
        // No instances (empty) or genuine error.
        return if ok(status) { Ok(Vec::new()) } else { Err(status) };
    }

    let mut buffer = vec![0u8; buffer_size as usize];
    let status = unsafe {
        PdhGetFormattedCounterArrayW(
            counter,
            PDH_FMT_LARGE,
            &mut buffer_size,
            &mut item_count,
            Some(buffer.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W),
        )
    };
    if !ok(status) {
        return Err(status);
    }

    let items = unsafe {
        std::slice::from_raw_parts(buffer.as_ptr() as *const PDH_FMT_COUNTERVALUE_ITEM_W, item_count as usize)
    };

    Ok(items
        .iter()
        .filter(|it| it.FmtValue.CStatus == 0)
        .map(|it| {
            let name = unsafe {
                it.szName.to_string().unwrap_or_default()
            };
            let value = unsafe { it.FmtValue.Anonymous.largeValue };
            (name, value)
        })
        .collect())
}

pub struct GpuPdhQuery {
    query: PDH_HQUERY,
    engine_counter: PDH_HCOUNTER,
    proc_mem_dedicated: PDH_HCOUNTER,
    proc_mem_shared: PDH_HCOUNTER,
    proc_mem_local: PDH_HCOUNTER,
    proc_mem_non_local: PDH_HCOUNTER,
    /// adapter-level dedicated commit (Task Manager "Dedicated GPU memory")
    adapter_dedicated: PDH_HCOUNTER,
    /// adapter-level resident VRAM (physical footprint)
    adapter_local: PDH_HCOUNTER,
    /// adapter-level resident shared (non-local) memory
    adapter_non_local: PDH_HCOUNTER,
}

/// Adapter-level memory sample (device-global accounting; the difference to
/// the per-process sums is kernel/driver/display-pipeline residency plus
/// evictable-but-committed segments).
/// Instance names: `luid_0x<high32>_0x<low32>_phys_N[_part_M]`.
pub struct AdapterMemSample {
    pub luid: (u32, u32),
    /// adapter dedicated commit, bytes
    pub dedicated_commit_bytes: u64,
    /// adapter resident VRAM, bytes
    pub local_resident_bytes: u64,
    /// adapter resident non-local, bytes
    pub non_local_resident_bytes: u64,
}

impl GpuPdhQuery {
    pub fn new() -> Result<Self, WinError> {
        let mut query = PDH_HQUERY::default();

        if !ok(unsafe { PdhOpenQueryW(None, 0, &mut query) }) {
            return Err(WinError::from_hresult(windows::core::HRESULT::from_win32(1)));
        }

        let mut engine_counter = PDH_HCOUNTER::default();
        let mut proc_mem_dedicated = PDH_HCOUNTER::default();
        let mut proc_mem_shared = PDH_HCOUNTER::default();
        let mut proc_mem_local = PDH_HCOUNTER::default();
        let mut proc_mem_non_local = PDH_HCOUNTER::default();
        let mut adapter_dedicated = PDH_HCOUNTER::default();
        let mut adapter_local = PDH_HCOUNTER::default();
        let mut adapter_non_local = PDH_HCOUNTER::default();

        let mut fail = 0;
        if !ok(unsafe {
            PdhAddEnglishCounterW(query, w!(r"\GPU Engine(*)\Utilization Percentage"), 0, &mut engine_counter)
        }) { fail = 2; }
        if !ok(unsafe {
            PdhAddEnglishCounterW(query, w!(r"\GPU Process Memory(*)\Dedicated Usage"), 0, &mut proc_mem_dedicated)
        }) { fail = 3; }
        if !ok(unsafe {
            PdhAddEnglishCounterW(query, w!(r"\GPU Process Memory(*)\Shared Usage"), 0, &mut proc_mem_shared)
        }) { fail = 4; }
        if !ok(unsafe {
            PdhAddEnglishCounterW(query, w!(r"\GPU Process Memory(*)\Local Usage"), 0, &mut proc_mem_local)
        }) { fail = 5; }
        if !ok(unsafe {
            PdhAddEnglishCounterW(query, w!(r"\GPU Process Memory(*)\Non Local Usage"), 0, &mut proc_mem_non_local)
        }) { fail = 6; }
        if !ok(unsafe {
            PdhAddEnglishCounterW(query, w!(r"\GPU Adapter Memory(*)\Dedicated Usage"), 0, &mut adapter_dedicated)
        }) { fail = 7; }
        if !ok(unsafe {
            PdhAddEnglishCounterW(query, w!(r"\GPU Local Adapter Memory(*)\Local Usage"), 0, &mut adapter_local)
        }) { fail = 8; }
        if !ok(unsafe {
            PdhAddEnglishCounterW(query, w!(r"\GPU Non Local Adapter Memory(*)\Non Local Usage"), 0, &mut adapter_non_local)
        }) { fail = 9; }

        if fail != 0 {
            unsafe { PdhCloseQuery(query) };
            return Err(WinError::from_hresult(windows::core::HRESULT::from_win32(fail)));
        }

        Ok(Self {
            query,
            engine_counter,
            proc_mem_dedicated,
            proc_mem_shared,
            proc_mem_local,
            proc_mem_non_local,
            adapter_dedicated,
            adapter_local,
            adapter_non_local,
        })
    }

    /// Collect one PDH sample. Utilization is a rate counter and only yields
    /// values from the second collect onwards.
    pub fn collect(
        &mut self,
    ) -> Result<(Vec<EngineSample>, Vec<ProcMemSample>, Vec<AdapterMemSample>), u32> {
        let status = unsafe { PdhCollectQueryData(self.query) };
        if !ok(status) {
            return Err(status);
        }

        // GPU Engine: collapse multiple engine instances of the same engtype per pid.
        let mut collapsed: HashMap<(u32, String), EngineSample> = HashMap::new();
        for (name, v) in read_counter_array(self.engine_counter)? {
            let (pid, luid) = parse_instance(&name);
            let engtype = engtype_of(&name);
            let e = collapsed.entry((pid, engtype.clone())).or_insert(EngineSample {
                pid,
                luid,
                engtype,
                utilization: 0.0,
            });
            e.utilization += v as f64;
            if e.luid.is_none() { e.luid = luid; }
        }
        let engines = collapsed.into_values().collect();

        // GPU Process Memory (all instant counters, bytes)
        let mut per_pid: HashMap<u32, ProcMemSample> = HashMap::new();
        let mut feed = |counter: PDH_HCOUNTER, put: &dyn Fn(&mut ProcMemSample, u64)| -> Result<(), u32> {
            for (name, v) in read_counter_array(counter)? {
                let (pid, luid) = parse_instance(&name);
                let e = per_pid.entry(pid).or_insert_with(|| ProcMemSample {
                    pid,
                    luid,
                    dedicated_commit_bytes: 0,
                    shared_commit_bytes: 0,
                    local_resident_bytes: 0,
                    non_local_resident_bytes: 0,
                });
                put(e, v as u64);
                if e.luid.is_none() { e.luid = luid; }
            }
            Ok(())
        };

        feed(self.proc_mem_local, &|e, v| e.local_resident_bytes = v)?;
        feed(self.proc_mem_non_local, &|e, v| e.non_local_resident_bytes = v)?;
        feed(self.proc_mem_dedicated, &|e, v| e.dedicated_commit_bytes = v)?;
        feed(self.proc_mem_shared, &|e, v| e.shared_commit_bytes = v)?;

        let proc_mem: Vec<ProcMemSample> = per_pid.into_values().collect();

        // Adapter-level memory (dedicated commit / local resident / non-local resident).
        // Instances are luid-keyed with optional _part_N suffixes (linked adapters);
        // parts are summed into one sample per LUID.
        let mut per_luid: HashMap<(u32, u32), AdapterMemSample> = HashMap::new();
        let mut feed_adapter =
            |counter: PDH_HCOUNTER, put: &dyn Fn(&mut AdapterMemSample, u64)| -> Result<(), u32> {
                for (name, v) in read_counter_array(counter)? {
                    let (pid, luid) = parse_instance(&name);
                    // adapter instance names have no pid segment (parse yields 0)
                    let Some(luid) = luid else { continue };
                    let e = per_luid.entry(luid).or_insert_with(|| AdapterMemSample {
                        luid,
                        dedicated_commit_bytes: 0,
                        local_resident_bytes: 0,
                        non_local_resident_bytes: 0,
                    });
                    put(e, v as u64);
                    let _ = pid;
                }
                Ok(())
            };

        feed_adapter(self.adapter_dedicated, &|e, v| {
            e.dedicated_commit_bytes = v.max(e.dedicated_commit_bytes)
        })?;
        feed_adapter(self.adapter_local, &|e, v| e.local_resident_bytes += v)?;
        feed_adapter(self.adapter_non_local, &|e, v| {
            e.non_local_resident_bytes += v
        })?;

        let adapter_mem: Vec<AdapterMemSample> = per_luid.into_values().collect();

        Ok((engines, proc_mem, adapter_mem))
    }
}

impl Drop for GpuPdhQuery {
    fn drop(&mut self) {
        if !self.query.0.is_null() {
            unsafe { PdhCloseQuery(self.query) };
        }
    }
}
