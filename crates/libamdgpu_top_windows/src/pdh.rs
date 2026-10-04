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
    /// dedicated (VRAM) commit, bytes — PDH "GPU Process Memory" values are bytes
    pub dedicated_bytes: u64,
    pub shared_bytes: u64,
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

        if fail != 0 {
            unsafe { PdhCloseQuery(query) };
            return Err(WinError::from_hresult(windows::core::HRESULT::from_win32(fail)));
        }

        Ok(Self { query, engine_counter, proc_mem_dedicated, proc_mem_shared })
    }

    /// Collect one PDH sample. Utilization is a rate counter and only yields
    /// values from the second collect onwards.
    pub fn collect(&mut self) -> Result<(Vec<EngineSample>, Vec<ProcMemSample>), u32> {
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

        // GPU Process Memory
        let mut dedicated: HashMap<u32, (Option<(u32, u32)>, u64)> = HashMap::new();
        for (name, v) in read_counter_array(self.proc_mem_dedicated)? {
            let (pid, luid) = parse_instance(&name);
            let e = dedicated.entry(pid).or_insert((luid, 0));
            e.1 = v as u64;
            if e.0.is_none() { e.0 = luid; }
        }

        let mut shared: HashMap<u32, u64> = HashMap::new();
        for (name, v) in read_counter_array(self.proc_mem_shared)? {
            let (pid, _) = parse_instance(&name);
            shared.insert(pid, v as u64);
        }

        let proc_mem = dedicated
            .into_iter()
            .map(|(pid, (luid, dedicated_bytes))| ProcMemSample {
                pid,
                luid,
                dedicated_bytes,
                shared_bytes: shared.get(&pid).copied().unwrap_or(0),
            })
            .collect();

        Ok((engines, proc_mem))
    }
}

impl Drop for GpuPdhQuery {
    fn drop(&mut self) {
        if !self.query.0.is_null() {
            unsafe { PdhCloseQuery(self.query) };
        }
    }
}
