//! libamdgpu_top_windows: Windows data-sampling backend for amdgpu_top (P0).
//!
//! Data source mapping (Linux -> Windows):
//! - `/dev/dri` device enumeration      -> DXGI `EnumAdapters1` (AMD vendor id 0x1002)
//! - drm fdinfo `drm-engine-*` (ns)     -> PDH `GPU Engine(*)\Utilization Percentage`
//! - drm fdinfo `drm-memory-vram/gtt`   -> PDH `GPU Process Memory(*)\Dedicated|Shared Usage`
//! - DRM ioctl `AMDGPU_INFO_MEMORY`     -> `DXGI_ADAPTER_DESC1` totals
//!
//! PDH is the same data source the Windows Task Manager "Performance/GPU" and
//! "Details" GPU columns use. Rate counters (utilization) need at least two
//! `PdhCollectQueryData` samples spaced by the update interval.

#![cfg(windows)]

mod dxgi;
mod pdh;
mod proc;

use serde::Serialize;
use std::collections::HashMap;
use std::time::Duration;

pub use dxgi::WinGpuAdapter;

/// Per-engine utilization of one process, in %.
/// Field names align with amdgpu_top's `FdInfoUsage` engine fields.
#[derive(Debug, Default, Clone, Serialize)]
pub struct EngineUsage {
    /// engtype_3D
    pub gfx: f64,
    /// engtype_Compute
    pub compute: f64,
    /// engtype_Copy
    pub dma: f64,
    /// engtype_VideoDecode
    pub dec: f64,
    /// engtype_VideoEncode
    pub enc: f64,
    /// engtype_VideoProcessing
    pub media: f64,
    /// any other/unknown engtype
    pub other: f64,
}

impl EngineUsage {
    pub fn total(&self) -> f64 {
        self.gfx + self.compute + self.dma + self.dec + self.enc + self.media + self.other
    }

    fn add(&mut self, engtype: &str, v: f64) {
        match engtype {
            "3D" => self.gfx += v,
            "Compute" => self.compute += v,
            "Copy" => self.dma += v,
            "VideoDecode" => self.dec += v,
            "VideoEncode" => self.enc += v,
            "VideoProcessing" => self.media += v,
            _ => self.other += v,
        }
    }
}

/// One process row: equivalent of amdgpu_top's `ProcInfo` + `FdInfoUsage`.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ProcGpuUsage {
    pub pid: u32,
    pub name: String,
    pub usage: EngineUsage,       // %
    /// committed dedicated (VRAM) memory, bytes — ≈ vram_usage
    pub dedicated_bytes: u64,
    /// committed shared memory, bytes — ≈ gtt_usage
    pub shared_bytes: u64,
}

/// Snapshot of one AMD adapter.
#[derive(Debug, Clone, Serialize)]
pub struct AdapterSnapshot {
    pub description: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// `{HighPart:0x..}_{LowPart:0x..}`
    pub luid: String,
    pub vram_total_kib: u64,
    pub shared_total_kib: u64,
    pub processes: Vec<ProcGpuUsage>,
    pub total_usage: EngineUsage,
    /// sum of per-process dedicated bytes
    pub dedicated_used_bytes: u64,
    pub shared_used_bytes: u64,
}

pub struct Sampler {
    adapters: Vec<WinGpuAdapter>,
    pdh: pdh::GpuPdhQuery,
    collected_once: bool,
}

pub struct SnapshotArgs {
    pub filter_pid: Option<u32>,
}

impl Sampler {
    pub fn new() -> windows::core::Result<Self> {
        Ok(Self {
            adapters: dxgi::enumerate_amd_adapters()?,
            pdh: pdh::GpuPdhQuery::new()?,
            collected_once: false,
        })
    }

    /// Prime the PDH rate counters so the first real snapshot has utilization.
    /// Call this once after `new()`, then wait >= interval before `snapshot()`.
    pub fn prime(&mut self) {
        let _ = self.pdh.collect();
        self.collected_once = true;
    }

    /// Collect one sample. Rate counters only become valid from the second
    /// call (PDH needs two samples spaced in time), so callers should poll
    /// this every `interval`.
    pub fn snapshot(&mut self, _interval: Duration, args: &SnapshotArgs) -> Vec<AdapterSnapshot> {
        let (engines, proc_mem) = match self.pdh.collect() {
            Ok(v) => v,
            Err(e) if !self.collected_once => {
                // First collect never has rates yet; ignore.
                let _ = e;
                (Vec::new(), Vec::new())
            },
            Err(_) => (Vec::new(), Vec::new()),
        };
        self.collected_once = true;

        let mut proc_names = proc::get_process_names();

        // (luid_high, luid_low) -> (pid -> ProcGpuUsage)
        type PidMap = HashMap<u32, ProcGpuUsage>;
        let mut per_adapter: HashMap<(u32, u32), PidMap> = HashMap::new();
        let mut luid_of_pid: HashMap<u32, (u32, u32)> = HashMap::new();

        for pdh::EngineSample { pid, luid, engtype, utilization } in engines {
            let Some(luid) = luid else { continue };
            let entry = per_adapter
                .entry(luid)
                .or_default()
                .entry(pid)
                .or_insert_with(|| ProcGpuUsage { pid, ..Default::default() });
            entry.usage.add(&engtype, utilization);
            luid_of_pid.insert(pid, luid);
        }

        for pdh::ProcMemSample { pid, luid, dedicated_bytes, shared_bytes } in proc_mem {
            let Some(luid) = luid else { continue };
            let entry = per_adapter
                .entry(luid)
                .or_default()
                .entry(pid)
                .or_insert_with(|| ProcGpuUsage { pid, ..Default::default() });
            entry.dedicated_bytes = dedicated_bytes;
            entry.shared_bytes = shared_bytes;
            luid_of_pid.insert(pid, luid);
        }

        self.adapters
            .iter()
            .map(|a| {
                let key = (a.luid_low, a.luid_high as u32);
                let pids = per_adapter.get(&key).cloned().unwrap_or_default();
                let mut processes: Vec<ProcGpuUsage> = pids
                    .into_values()
                    .filter(|p| args.filter_pid.is_none_or(|pid| pid == p.pid))
                    .map(|mut p| {
                        if p.name.is_empty() {
                            p.name = proc_names.remove(&p.pid).unwrap_or_else(|| "<unknown>".into());
                        }
                        p
                    })
                    .collect();
                processes.sort_by(|a, b| {
                    b.usage.total().partial_cmp(&a.usage.total())
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.dedicated_bytes.cmp(&a.dedicated_bytes))
                });

                let mut total_usage = EngineUsage::default();
                let mut dedicated_used = 0u64;
                let mut shared_used = 0u64;
                for p in &processes {
                    total_usage.gfx += p.usage.gfx;
                    total_usage.compute += p.usage.compute;
                    total_usage.dma += p.usage.dma;
                    total_usage.dec += p.usage.dec;
                    total_usage.enc += p.usage.enc;
                    total_usage.media += p.usage.media;
                    total_usage.other += p.usage.other;
                    dedicated_used += p.dedicated_bytes;
                    shared_used += p.shared_bytes;
                }

                AdapterSnapshot {
                    description: a.description.clone(),
                    vendor_id: a.vendor_id,
                    device_id: a.device_id,
                    luid: format!("0x{:08X}_0x{:08X}", a.luid_high as u32, a.luid_low),
                    vram_total_kib: a.dedicated_video_memory / 1024,
                    shared_total_kib: a.shared_system_memory / 1024,
                    processes,
                    total_usage,
                    dedicated_used_bytes: dedicated_used,
                    shared_used_bytes: shared_used,
                }
            })
            .collect()
    }
}
