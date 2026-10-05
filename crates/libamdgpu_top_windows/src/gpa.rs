//! GPA (GPU Performance API) SPM backend — GRBM/SQ-class hardware counters on
//! Windows via AMD's GPUPerfAPI (GPUOpen), the layer beneath RGP.
//!
//! SPM (Stream Performance Monitor) is timer-driven: after a
//! GpaSpmBegin/GpaSpmEnd command-list pair executes, the GPU samples the
//! enabled counter set every N SCLK cycles — hardware-global counts, like
//! reading GRBM on Linux, independent of which process is drawing.
//!
//! Runtime: GPUPerfAPIDX12-x64.dll from the GPA release zip. Located via
//! the AMDGPU_TOP_GPA_DLL env var, else probed next to the exe.
//!
//! Flow per sample:
//!   GpaBeginSession -> cmdlist: GpaSpmBegin -> submit -> <interval> ->
//!   cmdlist: GpaSpmEnd -> submit -> GpaEndSession ->
//!   GpaSpmGetSampleResult -> GpaSpmCalculateDerivedCounters (busy % series)

use std::ffi::{c_char, c_void, CStr};
use std::path::PathBuf;

use windows::core::{Interface, PCWSTR};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, ID3D12CommandAllocator, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12Resource, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_COMMAND_QUEUE_DESC, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
    D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET,
};
use windows::Win32::Graphics::Dxgi::IDXGIAdapter1;

// ---------------------------------------------------------------------------
// GPA C API (subset, all __cdecl exports of GPUPerfAPIDX12-x64.dll)
// ---------------------------------------------------------------------------

pub type GpaStatus = i32;
pub const GPA_STATUS_OK: GpaStatus = 0;

type GpaInitializeFn = unsafe extern "system" fn(u64) -> GpaStatus; // flags
type GpaDestroyFn = unsafe extern "system" fn() -> GpaStatus;
type GpaOpenContextFn = unsafe extern "system" fn(*mut c_void, u64, *mut *mut c_void) -> GpaStatus;
type GpaCreateSessionFn =
    unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> GpaStatus;
type GpaDeleteSessionFn = unsafe extern "system" fn(*mut c_void) -> GpaStatus;
type GpaBeginSessionFn = unsafe extern "system" fn(*mut c_void) -> GpaStatus;
type GpaEndSessionFn = unsafe extern "system" fn(*mut c_void) -> GpaStatus;
type GpaDisableAllCountersFn = unsafe extern "system" fn(*mut c_void) -> GpaStatus;
type GpaEnableCounterFn = unsafe extern "system" fn(*mut c_void, u32) -> GpaStatus;
type GpaBeginCommandListFn =
    unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32, *mut *mut c_void) -> GpaStatus;
type GpaEndCommandListFn = unsafe extern "system" fn(*mut c_void) -> GpaStatus;
type GpaBeginSampleFn = unsafe extern "system" fn(u32, *mut c_void) -> GpaStatus;
type GpaEndSampleFn = unsafe extern "system" fn(*mut c_void) -> GpaStatus;
type GpaGetSampleResultSizeFn =
    unsafe extern "system" fn(*mut c_void, u32, *mut usize) -> GpaStatus;
type GpaGetSampleResultFn =
    unsafe extern "system" fn(*mut c_void, u32, usize, *mut c_void) -> GpaStatus;
type GpaGetEnabledIndexFn = unsafe extern "system" fn(*mut c_void, u32, *mut u32) -> GpaStatus;
type GpaGetPassCountFn = unsafe extern "system" fn(*mut c_void, *mut u32) -> GpaStatus;
type GpaResetSessionFn = unsafe extern "system" fn(*mut c_void) -> GpaStatus;

/// kGpaCommandListPrimary
const GPA_CMDLIST_PRIMARY: u32 = 1;
type GpaGetNumCountersFn = unsafe extern "system" fn(*mut c_void, *mut u32) -> GpaStatus;
type GpaGetCounterNameFn =
    unsafe extern "system" fn(*mut c_void, u32, *mut *const c_char) -> GpaStatus;
type GpaGetNumEnabledCountersFn = unsafe extern "system" fn(*mut c_void, *mut u32) -> GpaStatus;
type GpaSpmSetSampleIntervalFn = unsafe extern "system" fn(*mut c_void, u32) -> GpaStatus;
type GpaGetStatusAsStrFn = unsafe extern "system" fn(GpaStatus) -> *const c_char;

#[repr(C)]
pub struct GpaSpmCounterInfo {
    pub gpu_block_id: u32,
    pub gpu_block_instance: u32,
    pub data_offset: u32,
    pub event_index: u32,
}

#[repr(C)]
pub struct GpaSpmData {
    pub number_of_timestamps: u32,
    pub number_of_spm_counter_info: u32,
    pub number_of_counter_data: u32,
    pub number_of_bytes_per_counter_data: u32,
    pub timestamps: *const u64,
    pub spm_counter_info: *const GpaSpmCounterInfo,
    /// counter-major layout: ctr_i at [i*n_timestamps .. (i+1)*n_timestamps)
    pub counter_data_16bit: *const u16,
}

type HModule = *mut c_void;
type FarProc = *mut c_void;

unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> HModule;
    fn GetProcAddress(module: HModule, name: *const c_char) -> FarProc;
    fn FreeLibrary(module: HModule) -> i32;
}

pub struct GpaApi {
    module: HModule,
    initialize: GpaInitializeFn,
    pub destroy: GpaDestroyFn,
    open_context: GpaOpenContextFn,
    create_session: GpaCreateSessionFn,
    delete_session: GpaDeleteSessionFn,
    begin_session: GpaBeginSessionFn,
    end_session: GpaEndSessionFn,
    disable_all_counters: GpaDisableAllCountersFn,
    enable_counter: GpaEnableCounterFn,
    begin_command_list: GpaBeginCommandListFn,
    end_command_list: GpaEndCommandListFn,
    begin_sample: GpaBeginSampleFn,
    end_sample: GpaEndSampleFn,
    get_sample_result_size: GpaGetSampleResultSizeFn,
    get_sample_result: GpaGetSampleResultFn,
    get_enabled_index: GpaGetEnabledIndexFn,
    get_pass_count: GpaGetPassCountFn,
    reset_session: GpaResetSessionFn,
    get_num_counters: GpaGetNumCountersFn,
    get_counter_name: GpaGetCounterNameFn,
    get_num_enabled: GpaGetNumEnabledCountersFn,
    spm_set_interval: GpaSpmSetSampleIntervalFn,
    status_as_str: GpaGetStatusAsStrFn,
}

fn dll_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("AMDGPU_TOP_GPA_DLL") {
        v.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        v.push(
            exe.parent()
                .unwrap_or(PathBuf::from(".").as_path())
                .join("GPUPerfAPIDX12-x64.dll"),
        );
    }
    v.push(PathBuf::from("GPUPerfAPIDX12-x64.dll"));
    v
}

impl GpaApi {
    pub fn load() -> Result<Self, String> {
        unsafe {
            let mut module: HModule = std::ptr::null_mut();
            let mut used = String::new();
            for path in dll_candidates() {
                let mut wide: Vec<u16> =
                    path.as_os_str().to_string_lossy().encode_utf16().collect();
                wide.push(0);
                module = LoadLibraryW(wide.as_ptr());
                if !module.is_null() {
                    used = path.display().to_string();
                    break;
                }
            }
            if module.is_null() {
                return Err("GPUPerfAPIDX12-x64.dll not found (set AMDGPU_TOP_GPA_DLL)".into());
            }

            macro_rules! sym {
                ($name:literal) => {{
                    let p = GetProcAddress(module, concat!($name, "\0").as_ptr() as *const c_char);
                    if p.is_null() {
                        FreeLibrary(module);
                        return Err(format!("missing export {}", $name));
                    }
                    p
                }};
            }

            let api = Self {
                module,
                initialize: std::mem::transmute::<FarProc, GpaInitializeFn>(sym!("GpaInitialize")),
                destroy: std::mem::transmute(sym!("GpaDestroy")),
                open_context: std::mem::transmute(sym!("GpaOpenContext")),
                create_session: std::mem::transmute(sym!("GpaCreateSession")),
                delete_session: std::mem::transmute(sym!("GpaDeleteSession")),
                begin_session: std::mem::transmute(sym!("GpaBeginSession")),
                end_session: std::mem::transmute(sym!("GpaEndSession")),
                disable_all_counters: std::mem::transmute(sym!("GpaDisableAllCounters")),
                enable_counter: std::mem::transmute(sym!("GpaEnableCounter")),
                begin_command_list: std::mem::transmute(sym!("GpaBeginCommandList")),
                end_command_list: std::mem::transmute(sym!("GpaEndCommandList")),
                begin_sample: std::mem::transmute(sym!("GpaBeginSample")),
                end_sample: std::mem::transmute(sym!("GpaEndSample")),
                get_sample_result_size: std::mem::transmute(sym!("GpaGetSampleResultSize")),
                get_sample_result: std::mem::transmute(sym!("GpaGetSampleResult")),
                get_enabled_index: std::mem::transmute(sym!("GpaGetEnabledIndex")),
                get_pass_count: std::mem::transmute(sym!("GpaGetPassCount")),
                reset_session: std::mem::transmute(sym!("GpaResetSession")),
                get_num_counters: std::mem::transmute(sym!("GpaGetNumCounters")),
                get_counter_name: std::mem::transmute(sym!("GpaGetCounterName")),
                get_num_enabled: std::mem::transmute(sym!("GpaGetNumEnabledCounters")),
                spm_set_interval: std::mem::transmute(sym!("GpaSpmSetSampleInterval")),
                status_as_str: std::mem::transmute(sym!("GpaGetStatusAsStr")),
            };

            let st = (api.initialize)(0);
            if st != GPA_STATUS_OK {
                let msg = api.status_str(st);
                FreeLibrary(module);
                return Err(format!("GpaInitialize failed: {msg}"));
            }
            eprintln!("[gpa] loaded: {used}");
            Ok(api)
        }
    }

    fn status_str(&self, st: GpaStatus) -> String {
        unsafe {
            let p = (self.status_as_str)(st);
            if p.is_null() {
                format!("{st}")
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    fn check(&self, st: GpaStatus, what: &str) -> Result<(), String> {
        if st == GPA_STATUS_OK {
            Ok(())
        } else {
            Err(format!("{what}: {}", self.status_str(st)))
        }
    }
}

impl Drop for GpaApi {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)();
            FreeLibrary(self.module);
        }
    }
}

// ---------------------------------------------------------------------------
// D3D12 device context to host the GPA session
// ---------------------------------------------------------------------------

struct D3d12Ctx {
    /// kept alive: command objects below keep refs, but hold the device
    #[allow(dead_code)]
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_value: u64,
    /// small render target + RTV for the sampling workload (Clear loop)
    /// kept alive while its RTV is in use
    #[allow(dead_code)]
    rt_resource: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    rtv_heap: Option<windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap>,
}

impl D3d12Ctx {
    fn new(adapter: IDXGIAdapter1) -> windows::core::Result<Self> {
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            let device = device.unwrap();

            let queue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;

            let allocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            // CreateCommandList returns an open (recording) list; close it so
            // that later Reset works (empty list, never submitted).
            list.Close()?;
            let fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;

            // Small render target + RTV for the sampling workload.
            let rtv_heap: windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })?;
            let heap_start = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            let mut rt_resource: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 1024,
                    Height: 1024,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut rt_resource,
            )?;
            let rt_resource = rt_resource.unwrap();
            device.CreateRenderTargetView(&rt_resource, None, heap_start);

            Ok(Self {
                device,
                queue,
                allocator,
                list,
                fence,
                fence_value: 0,
                rt_resource: Some(rt_resource),
                rtv_heap: Some(rtv_heap),
            })
        }
    }

    fn rtv(&self) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE> {
        self.rtv_heap
            .as_ref()
            .map(|h| unsafe { h.GetCPUDescriptorHandleForHeapStart() })
    }

    /// Record with |f| on a fresh command list, submit, wait on fence.
    fn submit<F: FnOnce(&Self)>(&mut self, f: F) -> windows::core::Result<()> {
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            f(self);
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.fence_value += 1;
            self.queue.Signal(&self.fence, self.fence_value)?;
            while self.fence.GetCompletedValue() < self.fence_value {
                std::hint::spin_loop();
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// SPM sampler
// ---------------------------------------------------------------------------

pub struct SpmCounter {
    pub name: String,
    /// mean of the derived series (percent counters: value units per GPA)
    pub mean: f64,
    pub last: f64,
    pub samples: usize,
}

pub struct SpmSampler {
    api: GpaApi,
    ctx: D3d12Ctx,
    #[allow(dead_code)]
    context: *mut c_void,
    session: *mut c_void,
    counter_names: Vec<String>,
    enabled_indices: Vec<u32>,
    started_once: bool,
}

/// Busy-percentage public counters available on gfx10+ (names from GPA's
/// public counter tables; presence is queried at runtime).
pub const BUSY_COUNTERS: &[&str] = &[
    "GPUBusy",
    "TessellatorBusy",
    "VsGsBusy",
    "PreTessellationBusy",
    "PostTessellationBusy",
    "PSBusy",
    "CSBusy",
    "PrimitiveAssemblyBusy",
    "DepthStencilTestBusy",
    "TexUnitBusy",
    "MemUnitBusy",
];

impl SpmSampler {
    pub fn new(
        adapter: IDXGIAdapter1,
        interval_sclk: u32,
        counter_names: &[String],
    ) -> Result<Self, String> {
        // D3D12 context first: when this runs inside a process that already
        // hosts a graphics stack (e.g. the eframe/wgpu GUI), creating the
        // device after GpaInitialize can fail with device-removed (0x887A0005).
        let mut ctx = None;
        for attempt in 0..3 {
            match D3d12Ctx::new(adapter.clone()) {
                Ok(c) => {
                    ctx = Some(c);
                    break;
                }
                Err(e) if attempt < 2 => {
                    // transient (heavy GPU load / driver settling); retry
                    eprintln!(
                        "[gpa] D3D12 init attempt {} failed: {e}, retrying",
                        attempt + 1
                    );
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                Err(e) => return Err(format!("D3D12: {e}")),
            }
        }
        let Some(ctx) = ctx else {
            return Err("D3D12: context creation failed".into());
        };

        let api = GpaApi::load()?;

        unsafe {
            let mut context: *mut c_void = std::ptr::null_mut();
            api.check(
                (api.open_context)(ctx.queue.as_raw(), 0, &mut context),
                "GpaOpenContext",
            )?;

            // kGpaSessionSampleTypeDiscreteCounter = 0. Officially only discrete
            // sessions are supported (streaming/SPM hangs at GpaSpmBegin on this
            // driver). One discrete sample is stretched across two submissions so
            // its counter window spans wall-clock time; hardware counters are
            // global, capturing whole-GPU activity.
            let mut session: *mut c_void = std::ptr::null_mut();
            api.check(
                (api.create_session)(context, 0, &mut session),
                "GpaCreateSession",
            )?;

            api.check((api.disable_all_counters)(session), "GpaDisableAllCounters")?;

            let mut name_to_index = std::collections::HashMap::new();
            let mut n_counters = 0u32;
            api.check(
                (api.get_num_counters)(session, &mut n_counters),
                "GpaGetNumCounters",
            )?;
            for i in 0..n_counters {
                let mut nm: *const c_char = std::ptr::null();
                if (api.get_counter_name)(session, i, &mut nm) == GPA_STATUS_OK && !nm.is_null() {
                    let s = CStr::from_ptr(nm).to_string_lossy().into_owned();
                    name_to_index.insert(s, i);
                }
            }

            let mut enabled = Vec::new();
            let mut enabled_indices = Vec::new();
            for name in counter_names {
                let Some(&index) = name_to_index.get(name) else {
                    eprintln!("[gpa] counter not present: {name}");
                    continue;
                };
                let st = (api.enable_counter)(session, index);
                if st == GPA_STATUS_OK {
                    enabled.push(name.clone());
                    enabled_indices.push(index);
                } else {
                    eprintln!("[gpa] enable {name} failed: {}", api.status_str(st));
                }
            }

            let mut num_enabled = 0u32;
            api.check(
                (api.get_num_enabled)(session, &mut num_enabled),
                "GpaGetNumEnabledCounters",
            )?;
            if num_enabled == 0 {
                return Err("no counters enabled".into());
            }

            // interval in SCLK cycles, 32..=4096 (driver default 4096)
            let interval = interval_sclk.clamp(32, 4096);
            let interval = if interval == 4096 {
                // GpaSpmSetSampleInterval deadlocks on this driver/dll combo;
                // 4096 is the documented default, so skip the call.
                eprintln!("[gpa] using default interval (skipping setter)");
                interval
            } else {
                api.check(
                    (api.spm_set_interval)(session, interval),
                    "GpaSpmSetSampleInterval",
                )?;
                eprintln!("[gpa] interval ok");
                interval
            };

            eprintln!("[gpa] SPM session: {num_enabled} counters, interval {interval} SCLK");

            Ok(Self {
                api,
                ctx,
                context,
                session,
                counter_names: enabled,
                enabled_indices,
                started_once: false,
            })
        }
    }

    /// Run one capture, returning per-counter values.
    ///
    /// Officially-supported discrete flow: one sample wraps a small built-in
    /// GPU workload (a Clear-render-target loop) inside a single command list:
    ///   begin_session
    ///     [cl] begin_command_list > begin_sample(0)
    ///          > OMSetRenderTargets + Clear x N   (sampling window)
    ///          > end_sample > end_command_list > submit
    ///   end_session > GpaGetSampleResult
    ///
    /// The hardware counters are device-global, so while our own workload
    /// keeps the counter window open, other processes' activity also
    /// accumulates into the counts (verified: CSBusy rises when an unrelated
    /// compute workload runs, though this process has no shaders at all).
    pub fn collect(&mut self, clears: u32) -> Result<Vec<SpmCounter>, String> {
        unsafe {
            const SAMPLE_ID: u32 = 0;

            if self.started_once {
                // A session cannot be re-begun; reset for the next capture.
                // GpaResetSession also clears the enabled counter set.
                self.api
                    .check((self.api.reset_session)(self.session), "GpaResetSession")?;
                self.api.check(
                    (self.api.disable_all_counters)(self.session),
                    "GpaDisableAllCounters",
                )?;
                for &index in &self.enabled_indices {
                    let st = ((&self.api).enable_counter)(self.session, index);
                    if st != GPA_STATUS_OK {
                        return Err(format!(
                            "re-enable counter {index}: {}",
                            self.api.status_str(st)
                        ));
                    }
                }
            }
            self.started_once = true;

            self.api
                .check((self.api.begin_session)(self.session), "GpaBeginSession")?;

            let session = self.session;
            let api = &self.api;

            // Multiple counters may require multiple passes; the sample must
            // exist in every pass (GPA multi-pass scheduling).
            let mut n_passes = 1u32;
            self.api.check(
                (self.api.get_pass_count)(session, &mut n_passes),
                "GpaGetPassCount",
            )?;
            eprintln!("[gpa] passes required: {n_passes}");

            for pass in 0..n_passes {
                let mut workload: Result<(), String> = Ok(());
                self.ctx
                    .submit(|ctx| {
                        workload = (|| -> Result<(), String> {
                            let mut cl: *mut c_void = std::ptr::null_mut();
                            api.check(
                                (api.begin_command_list)(
                                    session,
                                    pass,
                                    ctx.list.as_raw(),
                                    GPA_CMDLIST_PRIMARY,
                                    &mut cl,
                                ),
                                "GpaBeginCommandList",
                            )?;
                            api.check((api.begin_sample)(SAMPLE_ID, cl), "GpaBeginSample")?;

                            // Sampling workload: Clear loop on a 1024x1024 RT.
                            if let Some(rtv) = ctx.rtv() {
                                ctx.list.OMSetRenderTargets(1, Some(&rtv), false, None);
                                let color = [0.5f32, 0.5, 0.5, 1.0];
                                for _ in 0..clears {
                                    ctx.list.ClearRenderTargetView(rtv, &color, None);
                                }
                            }

                            api.check((api.end_sample)(cl), "GpaEndSample")?;
                            api.check((api.end_command_list)(cl), "GpaEndCommandList")?;
                            Ok(())
                        })();
                    })
                    .map_err(|e| format!("submit workload: {e}"))?;
                workload?;
            }

            self.api
                .check((self.api.end_session)(self.session), "GpaEndSession")?;

            self.read_derived(SAMPLE_ID)
        }
    }

    /// Read the discrete sample result: one 64-bit slot per enabled counter
    /// (percent counters are float64). Slots follow the enabled-index order.
    fn read_derived(&mut self, sample_id: u32) -> Result<Vec<SpmCounter>, String> {
        unsafe {
            let mut size = 0usize;
            self.api.check(
                (self.api.get_sample_result_size)(self.session, sample_id, &mut size),
                "GpaGetSampleResultSize",
            )?;
            let n_enabled = self.counter_names.len();
            if size == 0 || n_enabled == 0 {
                return Ok(Vec::new());
            }
            let n_slots = size / 8;

            let mut results = vec![0u64; n_slots];
            self.api.check(
                (self.api.get_sample_result)(
                    self.session,
                    sample_id,
                    size,
                    results.as_mut_ptr() as *mut c_void,
                ),
                "GpaGetSampleResult",
            )?;

            // Map result slot -> enabled counter index -> our counter name.
            let mut index_of: Vec<u32> = Vec::with_capacity(n_enabled);
            for i in 0..n_enabled as u32 {
                let mut idx = 0u32;
                self.api.check(
                    (self.api.get_enabled_index)(self.session, i, &mut idx),
                    "GpaGetEnabledIndex",
                )?;
                index_of.push(idx);
            }
            let mut name_by_index = std::collections::HashMap::new();
            {
                let mut n = 0u32;
                if (self.api.get_num_counters)(self.session, &mut n) == GPA_STATUS_OK {
                    for i in 0..n {
                        let mut nm: *const c_char = std::ptr::null();
                        if (self.api.get_counter_name)(self.session, i, &mut nm) == GPA_STATUS_OK
                            && !nm.is_null()
                        {
                            let s = CStr::from_ptr(nm).to_string_lossy().into_owned();
                            name_by_index.insert(i, s);
                        }
                    }
                }
            }

            Ok(index_of
                .into_iter()
                .zip(results.iter())
                .filter_map(|(idx, &raw)| {
                    let name = name_by_index.get(&idx)?;
                    let value = f64::from_bits(raw);
                    Some(SpmCounter {
                        name: name.clone(),
                        mean: value,
                        last: value,
                        samples: 1,
                    })
                })
                .collect())
        }
    }

    /// List public counter names (diagnostics).
    pub fn list_counters(&mut self, max: usize) -> Vec<String> {
        unsafe {
            let mut n = 0u32;
            if (self.api.get_num_counters)(self.session, &mut n) != GPA_STATUS_OK {
                return Vec::new();
            }
            let mut out = Vec::new();
            for i in 0..n.min(max as u32) {
                let mut p: *const c_char = std::ptr::null();
                if (self.api.get_counter_name)(self.session, i, &mut p) == GPA_STATUS_OK
                    && !p.is_null()
                {
                    out.push(CStr::from_ptr(p).to_string_lossy().into_owned());
                }
            }
            out
        }
    }
}

impl Drop for SpmSampler {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                (self.api.delete_session)(self.session);
            }
            // GpaCloseContext via api.destroy (Destroy closes all contexts).
        }
    }
}

#[allow(dead_code)]
fn _unused_pcwstr(p: PCWSTR) -> PCWSTR {
    p
}
