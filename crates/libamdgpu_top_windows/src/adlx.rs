//! ADLX sensor backend (P2): hwmon equivalent on Windows via AMD's ADLX SDK.
//!
//! Uses the official `amd-adlx` Rust bindings (GPUOpen ADLX repo, MIT).
//! Runtime: `amdadlx64.dll` ships with the AMD Adrenalin driver in System32.
//!
//! Call chain (mirrors Samples/rust/PerfGPUMetrics):
//! AdlxHelper -> IADLXSystem::GetPerformanceMonitoringServices
//!            -> IADLXSystem::GetGPUs -> IADLXGPUList
//!            -> GetSupportedGPUMetrics(gpu)  [cached support queries]
//!            -> GetCurrentGPUMetrics(gpu)    [per-tick refresh]
//!
//! ADLX GPUs are matched to DXGI adapters by PCI device id (DeviceId() hex
//! string, e.g. "7550"). Identical dual-GPU setups fall back to index order.

use amd_adlx::*;

use crate::SensorSnapshot;

/// Static device identity (ADLX IADLXGPU getters) — fills the Device Info
/// side panel. Fields unavailable in the same spirit as the Linux GUI
/// (chip class, CU counts, caches, gfx_target_version) have no Windows source.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct AdlxDeviceInfo {
    pub name: String,
    /// hex string, e.g. "7550"
    pub device_id: String,
    /// hex string, e.g. "C0"
    pub revision_id: String,
    pub subsystem_id: String,
    pub subsystem_vendor_id: String,
    pub unique_id: i32,
    pub total_vram_mb: u32,
    /// e.g. "GDDR6"
    pub vram_type: String,
    pub vbios_pn: String,
    pub vbios_version: String,
    pub vbios_date: String,
}

/// Per-metric supported ranges (ADLX GPUMetricsSupport getters) — used by the
/// sensor plots for y-axis bounds, mirroring the Linux GUI's min/max labels.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct SensorRange {
    pub gpu_usage: Option<(f64, f64)>,
    pub gpu_clock_mhz: Option<(f64, f64)>,
    pub vram_clock_mhz: Option<(f64, f64)>,
    pub temp_edge_c: Option<(f64, f64)>,
    pub temp_hotspot_c: Option<(f64, f64)>,
    pub fan_rpm: Option<(f64, f64)>,
    pub power_w: Option<(f64, f64)>,
    pub total_board_power_w: Option<(f64, f64)>,
    pub voltage_mv: Option<(f64, f64)>,
    pub vram_mb: Option<(f64, f64)>,
}

pub struct AdlxSensors {
    _helper: AdlxHelper,
    perf: *mut IADLXPerformanceMonitoringServices,
    gpus: Vec<AdlxGpu>,
}

struct AdlxGpu {
    gpu: *mut IADLXGPU,
    support: *mut IADLXGPUMetricsSupport,
    /// PCI device id (hex, e.g. 0x7550); kept for future exact matching —
    /// identical dual-GPU setups share it, so current pairing is by order.
    #[allow(dead_code)]
    device_id: u32,
    device_info: AdlxDeviceInfo,
    range: SensorRange,
}

fn cstr_ptr_to_string(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
}

impl AdlxSensors {
    /// Returns None when the ADLX runtime is unavailable (no AMD driver) —
    /// callers should degrade gracefully.
    pub fn new() -> Option<Self> {
        unsafe {
            let helper = AdlxHelper::new().ok()?;

            let sys = helper.system_services();
            let sys_vtbl = &*(*sys).pVtbl;

            let mut perf: *mut IADLXPerformanceMonitoringServices = std::ptr::null_mut();
            if !adlx_succeeded((sys_vtbl.GetPerformanceMonitoringServices)(sys, &mut perf)) {
                return None;
            }

            let mut gpus: *mut IADLXGPUList = std::ptr::null_mut();
            if !adlx_succeeded((sys_vtbl.GetGPUs)(sys, &mut gpus)) {
                ((*(*perf).pVtbl).Release)(perf);
                return None;
            }

            let list_vtbl = &*(*gpus).pVtbl;
            let size = (list_vtbl.Size)(gpus);
            let perf_vtbl = &*(*perf).pVtbl;

            let mut out = Vec::with_capacity(size as usize);

            for i in 0..size {
                let mut gpu: *mut IADLXGPU = std::ptr::null_mut();
                if !adlx_succeeded((list_vtbl.At_GPUList)(gpus, i, &mut gpu)) {
                    continue;
                }

                // PCI device id (hex string) for DXGI matching.
                let mut dev_cstr: *const std::os::raw::c_char = std::ptr::null();
                let device_id = if adlx_succeeded(((*(*gpu).pVtbl).DeviceId)(gpu, &mut dev_cstr)) {
                    u32::from_str_radix(&cstr_ptr_to_string(dev_cstr), 16).unwrap_or(u32::MAX)
                } else {
                    u32::MAX
                };

                let device_info = read_device_info(gpu);

                let mut support: *mut IADLXGPUMetricsSupport = std::ptr::null_mut();
                if adlx_succeeded((perf_vtbl.GetSupportedGPUMetrics)(perf, gpu, &mut support)) {
                    let range = read_ranges(support);
                    // Hold the At() reference for our lifetime (released in Drop).
                    out.push(AdlxGpu {
                        gpu,
                        support,
                        device_id,
                        device_info,
                        range,
                    });
                } else {
                    ((*(*gpu).pVtbl).Release)(gpu);
                }
            }

            (list_vtbl.Release)(gpus);

            if out.is_empty() {
                (perf_vtbl.Release)(perf);
                return None;
            }

            Some(Self {
                _helper: helper,
                perf,
                gpus: out,
            })
        }
    }

    /// Static per-GPU identity, in enumeration order.
    pub fn device_infos(&self) -> Vec<AdlxDeviceInfo> {
        self.gpus.iter().map(|g| g.device_info.clone()).collect()
    }

    /// Static per-GPU metric ranges, in enumeration order.
    pub fn ranges(&self) -> Vec<SensorRange> {
        self.gpus.iter().map(|g| g.range).collect()
    }

    /// Read current metrics for every ADLX GPU, in ADLX enumeration order.
    ///
    /// GPUs are matched to DXGI adapters by index: identical dual-GPU setups
    /// share the same PCI device id, and DXGI may also expose driver shadow
    /// adapters that ADLX does not enumerate (observed: single physical GPU
    /// reported twice by DXGI, once by ADLX) — those get sensors: None.
    pub fn snapshot(&mut self) -> Vec<SensorSnapshot> {
        let mut out = Vec::with_capacity(self.gpus.len());

        unsafe {
            let perf_vtbl = &*(*self.perf).pVtbl;

            for g in &self.gpus {
                let mut metrics: *mut IADLXGPUMetrics = std::ptr::null_mut();
                if !adlx_succeeded((perf_vtbl.GetCurrentGPUMetrics)(
                    self.perf,
                    g.gpu,
                    &mut metrics,
                )) {
                    continue;
                }

                let m = &*(*metrics).pVtbl;
                let s = &*(*g.support).pVtbl;
                let mptr = metrics;
                let sptr = g.support;

                let mut snap = SensorSnapshot::default();
                let mut supported: adlx_bool = 0;

                macro_rules! read {
                    ($is_sup:ident, $getter:ident, $ty:ty, $field:ident) => {
                        if adlx_succeeded((s.$is_sup)(sptr, &mut supported)) && supported != 0 {
                            let mut v: $ty = Default::default();
                            if adlx_succeeded((m.$getter)(mptr, &mut v)) {
                                snap.$field = Some(v as f64);
                            }
                        }
                    };
                }

                read!(IsSupportedGPUUsage, GPUUsage, adlx_double, gpu_usage);
                read!(
                    IsSupportedGPUClockSpeed,
                    GPUClockSpeed,
                    adlx_int,
                    gpu_clock_mhz
                );
                read!(
                    IsSupportedGPUVRAMClockSpeed,
                    GPUVRAMClockSpeed,
                    adlx_int,
                    vram_clock_mhz
                );
                read!(
                    IsSupportedGPUTemperature,
                    GPUTemperature,
                    adlx_double,
                    temp_edge_c
                );
                read!(
                    IsSupportedGPUHotspotTemperature,
                    GPUHotspotTemperature,
                    adlx_double,
                    temp_hotspot_c
                );
                read!(
                    IsSupportedGPUIntakeTemperature,
                    GPUIntakeTemperature,
                    adlx_double,
                    temp_intake_c
                );
                read!(IsSupportedGPUPower, GPUPower, adlx_double, power_w);
                read!(
                    IsSupportedGPUTotalBoardPower,
                    GPUTotalBoardPower,
                    adlx_double,
                    total_board_power_w
                );
                read!(IsSupportedGPUFanSpeed, GPUFanSpeed, adlx_int, fan_rpm);
                read!(IsSupportedGPUVRAM, GPUVRAM, adlx_int, vram_used_mb);
                read!(IsSupportedGPUVoltage, GPUVoltage, adlx_int, voltage_mv);

                (m.Release)(metrics);

                out.push(snap);
            }
        }

        out
    }
}

/// Read static identity fields from an IADLXGPU.
fn read_device_info(gpu: *mut IADLXGPU) -> AdlxDeviceInfo {
    let mut info = AdlxDeviceInfo::default();
    let v = unsafe { &*(*gpu).pVtbl };

    macro_rules! cstr_field {
        ($getter:ident, $field:ident) => {{
            let mut p: *const std::os::raw::c_char = std::ptr::null();
            if adlx_succeeded((v.$getter)(gpu, &mut p)) {
                info.$field = cstr_ptr_to_string(p);
            }
        }};
    }

    unsafe {
        cstr_field!(Name, name);
        cstr_field!(DeviceId, device_id);
        cstr_field!(RevisionId, revision_id);
        cstr_field!(SubSystemId, subsystem_id);
        cstr_field!(SubSystemVendorId, subsystem_vendor_id);
        cstr_field!(VRAMType, vram_type);

        let mut vram = 0u32;
        if adlx_succeeded((v.TotalVRAM)(gpu, &mut vram)) {
            info.total_vram_mb = vram;
        }

        let mut uid = 0i32;
        if adlx_succeeded((v.UniqueId)(gpu, &mut uid)) {
            info.unique_id = uid;
        }

        let (mut pn, mut ver, mut date): (
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
        ) = (std::ptr::null(), std::ptr::null(), std::ptr::null());
        if adlx_succeeded((v.BIOSInfo)(gpu, &mut pn, &mut ver, &mut date)) {
            info.vbios_pn = cstr_ptr_to_string(pn);
            info.vbios_version = cstr_ptr_to_string(ver);
            info.vbios_date = cstr_ptr_to_string(date);
        }
    }

    info
}

/// Read supported metric ranges once (y-axis bounds for the sensor plots).
fn read_ranges(support: *mut IADLXGPUMetricsSupport) -> SensorRange {
    let mut r = SensorRange::default();
    let s = unsafe { &*(*support).pVtbl };

    macro_rules! range {
        ($getter:ident, $field:ident) => {{
            let (mut min, mut max) = (0i32, 0i32);
            if adlx_succeeded((s.$getter)(support, &mut min, &mut max)) {
                r.$field = Some((min as f64, max as f64));
            }
        }};
    }

    unsafe {
        range!(GetGPUUsageRange, gpu_usage);
        range!(GetGPUClockSpeedRange, gpu_clock_mhz);
        range!(GetGPUVRAMClockSpeedRange, vram_clock_mhz);
        range!(GetGPUTemperatureRange, temp_edge_c);
        range!(GetGPUHotspotTemperatureRange, temp_hotspot_c);
        range!(GetGPUPowerRange, power_w);
        range!(GetGPUTotalBoardPowerRange, total_board_power_w);
        range!(GetGPUFanSpeedRange, fan_rpm);
        range!(GetGPUVRAMRange, vram_mb);
        range!(GetGPUVoltageRange, voltage_mv);
    }

    r
}

impl Drop for AdlxSensors {
    fn drop(&mut self) {
        unsafe {
            for g in &self.gpus {
                ((*(*g.support).pVtbl).Release)(g.support);
                ((*(*g.gpu).pVtbl).Release)(g.gpu);
            }
            if !self.perf.is_null() {
                ((*(*self.perf).pVtbl).Release)(self.perf);
            }
        }
    }
}
