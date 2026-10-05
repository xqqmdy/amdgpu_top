//! amdgpu_top_win GUI (P3): eframe frontend for the Windows backend.
//!
//! Architecture: a worker thread owns the Sampler (PDH handles and ADLX COM
//! pointers are not Send), polls it every interval and ships owned snapshots
//! over an mpsc channel; the UI thread only renders pure data.
//!
//! Layout matches the Linux amdgpu_top GUI single-page style: device selector
//! on top, then all sections (Sensors, Processes, VRAM) stacked on one page.

#![cfg(windows)]
// Release builds hide the console window; debug keeps it for eprintln output.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use egui_plot::{Axis, AxisHints, Legend, Line, Plot};
use libamdgpu_top_windows::{AdapterSnapshot, EngineUsage, Sampler, SnapshotArgs};
#[cfg(any(feature = "adlx", feature = "gpa"))]
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "adlx")]
use libamdgpu_top_windows::SensorSnapshot;

const MAX_SAMPLES: usize = 240;

// Layout constants copied from the Linux GUI (crates/amdgpu_top_gui/src/app.rs).
const SPACING: [f32; 2] = [16.0; 2];
const SENSORS_HEIGHT: f32 = 96.0;
const SENSORS_WIDTH: f32 = SENSORS_HEIGHT * 4.0;
const FDINFO_LIST_HEIGHT: f32 = 208.0;
const PLOT_HEIGHT: f32 = 208.0;
const PLOT_WIDTH: f32 = PLOT_HEIGHT * 5.0;

// ---------------------------------------------------------------------------
// worker
// ---------------------------------------------------------------------------

enum Msg {
    Snap(Vec<AdapterSnapshot>),
    Fatal(String),
    #[cfg(feature = "gpa")]
    GpaInitFailed(String),
    #[cfg(feature = "gpa")]
    GpaCounters(Vec<libamdgpu_top_windows::gpa::SpmCounter>),
}

// ---------------------------------------------------------------------------
// app state
// ---------------------------------------------------------------------------

#[cfg(feature = "adlx")]
struct SampleRecord {
    idx: f64,
    s: Option<SensorSnapshot>,
}

/// Per-tick total engine usage sample (mirrors fdinfo_history).
struct EngineHistRecord {
    idx: f64,
    u: EngineUsage,
}

/// Collapsible section, Linux-GUI style.
fn collapsing(ui: &mut egui::Ui, title: &str, default_open: bool, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(title)
        .default_open(default_open)
        .show(ui, add);
}

/// One sensor mini-plot cell: title with live value + filled line, y-axis
/// pinned to the ADLX-supported range (usage range is 0-100, matching the
/// Linux GUI's include_y(0)/include_y(100) percent plots). Axes are shown
/// with the x axis formatted as relative seconds.
#[cfg_attr(not(any(feature = "adlx", feature = "gpa")), allow(dead_code))]
fn sensor_plot(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    val: Option<f64>,
    unit: &str,
    range: Option<(f64, f64)>,
    points: Vec<[f64; 2]>,
) {
    let (min, max) = range.unwrap_or((0.0, 100.0));

    egui::Grid::new(id).spacing(SPACING).show(ui, |ui| {
        match val {
            Some(v) => ui.label(format!("{label} ({v:4.0} {unit})")),
            None => ui.label(label),
        };
        ui.end_row();

        let line = egui_plot::Line::new(id, points).fill(0.0_f32);
        let hover_unit = unit.to_string();
        egui_plot::Plot::new(id)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .include_y(min)
            .include_y(max)
            .custom_x_axes(vec![time_x_axis()])
            .label_formatter(move |_, v| format!("{:.1}s\n{:.1} {hover_unit}", v.x, v.y))
            .height(SENSORS_HEIGHT)
            .width(SENSORS_WIDTH)
            .show(ui, |p| p.line(line));
    });
}

/// Shared x axis hints: ticks in relative seconds (plot x = seconds since
/// app start, like the Linux GUI's vec_plotpoint x).
fn time_x_axis() -> AxisHints<'static> {
    AxisHints::new(Axis::X).formatter(|mark, _| format!("{:.0}s", mark.value))
}

/// VRAM history sample (always collected, like the Linux GUI's vram_history)
struct VramRecord {
    idx: f64,
    /// Σ per-process resident local bytes (user-attributable subset)
    resident: u64,
    /// adapter-level dedicated commit (Task Manager / ADLX GPUVRAM semantics)
    device_commit: u64,
    /// adapter-level resident VRAM (physical footprint)
    device_resident: u64,
    /// adapter-level resident non-local bytes
    device_shared: u64,
    /// adapter totals, bytes
    vram_total: u64,
    shared_total: u64,
}

struct DeviceUi {
    description: String,
    device_id: u32,
    revision_id: u32,
    luid: String,
    last: Option<AdapterSnapshot>,
    /// plot x value: seconds since app start (shared time axis)
    sample_idx: f64,
    vram_history: std::collections::VecDeque<VramRecord>,
    engine_history: std::collections::VecDeque<EngineHistRecord>,
    #[cfg(feature = "adlx")]
    history: VecDeque<SampleRecord>,
}

impl DeviceUi {
    fn push_vram(&mut self, dev: &AdapterSnapshot) {
        self.vram_history.push_back(VramRecord {
            idx: self.sample_idx,
            resident: dev.vram_resident_used_bytes,
            device_commit: dev.vram_commit_device_bytes,
            device_resident: dev.vram_resident_device_bytes,
            device_shared: dev.shared_resident_device_bytes,
            vram_total: dev.vram_total_kib * 1024,
            shared_total: dev.shared_total_kib * 1024,
        });
        while self.vram_history.len() > MAX_SAMPLES {
            self.vram_history.pop_front();
        }
    }
}

struct App {
    rx: Receiver<Msg>,
    fatal: Option<String>,
    devices: Vec<DeviceUi>,
    selected: usize,
    interval_ms: Arc<AtomicU64>,
    /// shared time origin for plot x values (GPA history included)
    start: Instant,
    #[cfg(feature = "gpa")]
    gpa: GpaState,
}

/// GPUPerfAPI hardware counters (first AMD adapter; blocking slow sampler
/// runs on its own thread, so values arrive at their own pace).
#[cfg(feature = "gpa")]
enum GpaState {
    Waiting,
    InitFailed(String),
    Live {
        counters: Vec<libamdgpu_top_windows::gpa::SpmCounter>,
        history: VecDeque<CounterRecord>,
    },
}

/// One GPA capture sample; values are aligned with the counters vector
/// received alongside (per-counter mini-plot grid).
#[cfg(feature = "gpa")]
struct CounterRecord {
    idx: f64,
    values: Vec<f64>,
}

#[cfg(feature = "gpa")]
impl CounterRecord {
    fn from_counters(idx: f64, counters: &[libamdgpu_top_windows::gpa::SpmCounter]) -> Self {
        Self {
            idx,
            values: counters.iter().map(|c| c.mean).collect(),
        }
    }
}

impl App {
    fn new(rx: Receiver<Msg>, interval_ms: Arc<AtomicU64>) -> Self {
        Self {
            rx,
            fatal: None,
            devices: Vec::new(),
            selected: 0,
            interval_ms,
            start: Instant::now(),
            #[cfg(feature = "gpa")]
            gpa: GpaState::Waiting,
        }
    }

    fn poll(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Fatal(e) => self.fatal = Some(e),
                #[cfg(feature = "gpa")]
                Msg::GpaInitFailed(e) => {
                    if matches!(self.gpa, GpaState::Waiting) {
                        self.gpa = GpaState::InitFailed(e);
                    }
                }
                #[cfg(feature = "gpa")]
                Msg::GpaCounters(counters) => {
                    let mut history = match std::mem::replace(&mut self.gpa, GpaState::Waiting) {
                        GpaState::Live { history, .. } => history,
                        _ => VecDeque::new(),
                    };
                    let idx = self.start.elapsed().as_secs_f64();
                    history.push_back(CounterRecord::from_counters(idx, &counters));
                    while history.len() > MAX_SAMPLES {
                        history.pop_front();
                    }
                    self.gpa = GpaState::Live { counters, history };
                }
                Msg::Snap(snap) => {
                    let t = self.start.elapsed().as_secs_f64();
                    for (i, dev) in snap.iter().enumerate() {
                        if let Some(slot) = self.devices.get_mut(i) {
                            slot.description = dev.description.clone();
                            slot.device_id = dev.device_id;
                            slot.revision_id = dev.revision;
                            slot.luid = dev.luid.clone();
                            slot.sample_idx = t;
                            slot.push_vram(dev);
                            slot.engine_history.push_back(EngineHistRecord {
                                idx: slot.sample_idx,
                                u: dev.total_usage.clone(),
                            });
                            while slot.engine_history.len() > MAX_SAMPLES {
                                slot.engine_history.pop_front();
                            }
                            #[cfg(feature = "adlx")]
                            {
                                slot.history.push_back(SampleRecord {
                                    idx: slot.sample_idx,
                                    s: dev.sensors.clone(),
                                });
                                while slot.history.len() > MAX_SAMPLES {
                                    slot.history.pop_front();
                                }
                            }
                            slot.last = Some(dev.clone());
                        } else {
                            let mut slot = DeviceUi {
                                description: dev.description.clone(),
                                device_id: dev.device_id,
                                revision_id: dev.revision,
                                luid: dev.luid.clone(),
                                last: Some(dev.clone()),
                                sample_idx: t,
                                vram_history: std::collections::VecDeque::new(),
                                engine_history: std::collections::VecDeque::new(),
                                #[cfg(feature = "adlx")]
                                history: VecDeque::new(),
                            };
                            slot.push_vram(dev);
                            #[cfg(feature = "adlx")]
                            slot.history.push_back(SampleRecord {
                                idx: slot.sample_idx,
                                s: dev.sensors.clone(),
                            });
                            self.devices.push(slot);
                        }
                    }

                    if self.selected >= self.devices.len() {
                        self.selected = 0;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// plots helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// eframe App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        ctx.request_repaint_after(Duration::from_millis(
            self.interval_ms.load(Ordering::Relaxed).max(200) / 2,
        ));

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(f) = &self.fatal {
                    ui.colored_label(egui::Color32::RED, f);
                    return;
                }
                if self.devices.is_empty() {
                    ui.label("Waiting for first sample…");
                    return;
                }

                egui::ComboBox::from_id_salt("device")
                    .selected_text(
                        self.devices
                            .get(self.selected)
                            .map(|d| d.description.clone())
                            .unwrap_or_default(),
                    )
                    .width(240.0)
                    .show_ui(ui, |ui| {
                        for (i, d) in self.devices.iter().enumerate() {
                            ui.selectable_value(&mut self.selected, i, &d.description);
                        }
                    });

                let dev = &self.devices[self.selected];
                ui.label(
                    egui::RichText::new(format!("id 0x{:04X} · {}", dev.device_id, dev.luid))
                        .small()
                        .weak(),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut ms = self.interval_ms.load(Ordering::Relaxed);
                    ui.add(
                        egui::DragValue::new(&mut ms)
                            .range(100..=5000)
                            .suffix(" ms"),
                    );
                    self.interval_ms.store(ms, Ordering::Relaxed);
                });
            });
        });

        // Device Info side panel (Linux GUI equivalent, static fields).
        // Default width fits the longest row on one unwrapped line; the user
        // can still resize (egui then remembers the manual size). The panel
        // must not be shown before the first snapshot: egui stores the panel
        // size on every frame it is shown, so an early empty frame would pin
        // the fallback width in PanelState and shadow the fitted default.
        if let Some(dev) = self.devices.get(self.selected) {
            let panel_w = device_info_panel_width(ctx, dev);
            egui::SidePanel::left("device_info")
                .default_width(panel_w)
                .resizable(true)
                .show(ctx, |ui| device_info_section(ui, dev));
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(f) = &self.fatal {
                ui.colored_label(egui::Color32::RED, f);
                return;
            }
            let Some(dev) = self.devices.get(self.selected) else {
                ui.spinner();
                return;
            };
            let Some(snap) = &dev.last else { return };

            // single page, all sections stacked vertically (like the Linux GUI);
            // Sensors last (Linux GUI puts sensor plots after the process list)
            egui::ScrollArea::vertical().show(ui, |ui| {
                activity_section(ui, dev);
                #[cfg(feature = "gpa")]
                {
                    ui.add_space(8.0);
                    hardware_counters_section(ui, &self.gpa);
                }
                ui.add_space(8.0);
                vram_section(ui, dev, snap);
                ui.add_space(8.0);
                processes_section(ui, dev, snap);
                ui.add_space(8.0);
                sensors_section(ui, dev, snap);
            });
        });
    }
}

// ---------------------------------------------------------------------------
// sections (single page)
// ---------------------------------------------------------------------------

/// Sidebar width that keeps the longest Device Info row on a single
/// unwrapped line: the widest key+value pair measured with the body font,
/// plus item spacing, panel margins, collapsing-body indent, a vertical
/// scrollbar allowance and a small safety margin for the bold key labels.
fn device_info_panel_width(ctx: &egui::Context, dev: &DeviceUi) -> f32 {
    let style = ctx.style();
    let font = egui::TextStyle::Body.resolve(&style);
    let text_w = |s: &str| {
        ctx.fonts_mut(|f| {
            f.layout_no_wrap(s.to_owned(), font.clone(), egui::Color32::WHITE)
                .size()
                .x
        })
    };

    let mut w = 0.0_f32;
    let mut row = |k: &str, v: String| {
        w = w.max(text_w(k) + text_w(&v));
    };

    row("Device Name", dev.description.clone());
    row(
        "DID : RID",
        format!("0x{:04X} : 0x{:02X}", dev.device_id, dev.revision_id),
    );
    row("LUID", dev.luid.clone());

    #[cfg(feature = "adlx")]
    if let Some(info) = dev.last.as_ref().and_then(|s| s.device_info.as_ref()) {
        row(
            "SubSystem",
            format!(
                "0x{} (vendor 0x{})",
                info.subsystem_id, info.subsystem_vendor_id
            ),
        );
        row("Unique ID", info.unique_id.to_string());
        row(
            "VRAM",
            format!("{} MB ({})", info.total_vram_mb, info.vram_type),
        );
        row("VBIOS PN", info.vbios_pn.clone());
        row("VBIOS Ver", info.vbios_version.clone());
        row("VBIOS Date", info.vbios_date.clone());
    }

    let spacing = &style.spacing;
    w + spacing.item_spacing.x
        + spacing.indent
        + (spacing.window_margin.left + spacing.window_margin.right) as f32
        + 28.0
}

/// Device Info side panel — static identity (DXGI + ADLX where the
/// feature is enabled). Linux-GUI fields with no Windows source (chip class,
/// CU counts, caches, gfx_target_version, IP discovery) are omitted.
fn device_info_section(ui: &mut egui::Ui, dev: &DeviceUi) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        collapsing(ui, "Device Info", true, |ui| {
            // Wrapping rows instead of a Grid on purpose: a Grid sizes its
            // columns to the intrinsic (unwrapped) text width, so long values
            // (device name, VBIOS strings) kept the content rect wider than
            // the panel. On egui 0.33 the painted resize separator tracks the
            // content rect while the drag strip tracks the panel rect (fixed
            // upstream in egui #8056), which broke sidebar resizing.
            let row = |ui: &mut egui::Ui, k: &str, v: String| {
                ui.horizontal_wrapped(|ui| {
                    ui.strong(k);
                    ui.label(v);
                });
            };
            row(ui, "Device Name", dev.description.clone());
            row(
                ui,
                "DID : RID",
                format!("0x{:04X} : 0x{:02X}", dev.device_id, dev.revision_id),
            );
            row(ui, "LUID", dev.luid.clone());

            #[cfg(feature = "adlx")]
            if let Some(snap) = &dev.last {
                if let Some(info) = &snap.device_info {
                    row(
                        ui,
                        "SubSystem",
                        format!(
                            "0x{} (vendor 0x{})",
                            info.subsystem_id, info.subsystem_vendor_id
                        ),
                    );
                    row(ui, "Unique ID", info.unique_id.to_string());
                    row(
                        ui,
                        "VRAM",
                        format!("{} MB ({})", info.total_vram_mb, info.vram_type),
                    );
                    row(ui, "VBIOS PN", info.vbios_pn.clone());
                    row(ui, "VBIOS Ver", info.vbios_version.clone());
                    row(ui, "VBIOS Date", info.vbios_date.clone());
                }
            }
        });
    });
}

/// Sensors — per-metric mini-plots in a two-column grid, mirroring the
/// Linux GUI's style (label with live value + percent-in-range, filled line,
/// y-axis pinned to the ADLX-supported range). Temperature plots use the
/// range max as the critical reference.
#[cfg_attr(not(feature = "adlx"), allow(unused_variables))]
fn sensors_section(ui: &mut egui::Ui, dev: &DeviceUi, snap: &AdapterSnapshot) {
    collapsing(ui, "Sensors", true, |ui| {
        #[cfg(not(feature = "adlx"))]
        {
            ui.label(
                egui::RichText::new(
                    "ADLX sensor panel requires the 'adlx' feature:\n  cargo build -p amdgpu_top_gui_windows --features adlx",
                )
                .weak(),
            );
        }

        #[cfg(feature = "adlx")]
        {
            let Some(s) = &snap.sensors else {
                ui.label(
                    egui::RichText::new(
                        "ADLX sensors unavailable for this adapter (shadow adapter or driver without ADLX).",
                    )
                    .weak(),
                );
                return;
            };

            let hist = &dev.history;
            let series = |get: &dyn Fn(&SensorSnapshot) -> Option<f64>| -> Vec<[f64; 2]> {
                hist.iter()
                    .filter_map(|r| get(r.s.as_ref()?).map(|v| [r.idx, v]))
                    .collect()
            };

            // (grid slot id, label, live value, unit, range, series)
            let mut items: Vec<(
                &str,
                &str,
                Option<f64>,
                &str,
                Option<(f64, f64)>,
                Vec<[f64; 2]>,
            )> = Vec::new();
            let r = snap.sensors_range;

            items.push((
                "s_clk",
                "GPU Clock",
                s.gpu_clock_mhz,
                "MHz",
                r.as_ref().and_then(|r| r.gpu_clock_mhz),
                series(&|s| s.gpu_clock_mhz),
            ));
            items.push((
                "s_mclk",
                "VRAM Clock",
                s.vram_clock_mhz,
                "MHz",
                r.as_ref().and_then(|r| r.vram_clock_mhz),
                series(&|s| s.vram_clock_mhz),
            ));
            items.push((
                "s_volt",
                "Voltage",
                s.voltage_mv,
                "mV",
                r.as_ref().and_then(|r| r.voltage_mv),
                series(&|s| s.voltage_mv),
            ));
            items.push((
                "s_bpw",
                "Board Power",
                s.total_board_power_w,
                "W",
                r.as_ref().and_then(|r| r.total_board_power_w),
                series(&|s| s.total_board_power_w),
            ));
            items.push((
                "s_pw",
                "Chip Power",
                s.power_w,
                "W",
                r.as_ref().and_then(|r| r.power_w),
                series(&|s| s.power_w),
            ));
            items.push((
                "s_fan",
                "Fan",
                s.fan_rpm,
                "RPM",
                r.as_ref().and_then(|r| r.fan_rpm),
                series(&|s| s.fan_rpm),
            ));
            items.push((
                "s_usage",
                "GPU Usage",
                s.gpu_usage,
                "%",
                r.as_ref().and_then(|r| r.gpu_usage),
                series(&|s| s.gpu_usage),
            ));
            items.push((
                "s_vrammb",
                "VRAM Used",
                s.vram_used_mb,
                "MB",
                r.as_ref().and_then(|r| r.vram_mb),
                series(&|s| s.vram_used_mb),
            ));

            let mut n = 1usize;
            egui::Grid::new("sensors_grid")
                .num_columns(2)
                .show(ui, |ui| {
                    for (id, label, val, unit, range, pts) in items {
                        if val.is_none() && pts.is_empty() {
                            continue;
                        }
                        sensor_plot(ui, id, label, val, unit, range, pts);
                        if n % 2 == 0 {
                            ui.end_row();
                        }
                        n += 1;
                    }
                    if n % 2 == 0 {
                        ui.end_row();
                    }
                });

            // temperatures (range max == critical reference)
            ui.add_space(4.0);
            ui.label(egui::RichText::new("Temperatures").small().weak());
            let mut n = 1usize;
            egui::Grid::new("temp_grid").num_columns(2).show(ui, |ui| {
                for (id, label, val, range) in [
                    (
                        "t_edge",
                        "Edge Temp",
                        s.temp_edge_c,
                        r.as_ref().and_then(|r| r.temp_edge_c),
                    ),
                    (
                        "t_hot",
                        "Junction Temp",
                        s.temp_hotspot_c,
                        r.as_ref().and_then(|r| r.temp_hotspot_c),
                    ),
                    ("t_intake", "Intake Temp", s.temp_intake_c, None),
                ] {
                    if val.is_none() {
                        continue;
                    }
                    let pts = series(&match id {
                        "t_edge" => |s: &SensorSnapshot| s.temp_edge_c,
                        "t_hot" => |s: &SensorSnapshot| s.temp_hotspot_c,
                        _ => |s: &SensorSnapshot| s.temp_intake_c,
                    });
                    let rng = range.or(Some((0.0, 110.0)));
                    sensor_plot(ui, id, label, val, "C", rng, pts);
                    if n % 2 == 0 {
                        ui.end_row();
                    }
                    n += 1;
                }
                if n % 2 == 0 {
                    ui.end_row();
                }
            });
        }
    });
}

/// Hardware IP busy counters via GPUPerfAPI (first AMD adapter).
/// Values come from a slow blocking sampler thread (one capture takes
/// seconds: multi-pass + built-in Clear workload), so they refresh at
/// their own pace, independent of the PDH snapshot interval.
/// Hardware Counters (GPUPerfAPI) — per-counter mini-plots in a
/// two-column grid, mirroring the Linux GUI's GRBM/GRBM2 section style.
/// Values come from a slow blocking sampler thread (one capture takes
/// seconds: multi-pass + built-in Clear workload), so they refresh at
/// their own pace, independent of the PDH snapshot interval.
#[cfg(feature = "gpa")]
fn hardware_counters_section(ui: &mut egui::Ui, gpa: &GpaState) {
    collapsing(ui, "Hardware Counters (GPUPerfAPI)", true, |ui| {
        ui.label(
            egui::RichText::new(
                "device-global HW counters, first AMD adapter · window = built-in workload (self-load inflates GPU-busy-class counters) · pipeline-stage granularity, not GRBM IP bits",
            )
            .small()
            .weak(),
        );
        ui.add_space(4.0);

        match gpa {
            GpaState::Waiting => {
                ui.spinner();
            }
            GpaState::InitFailed(e) => {
                ui.label(
                    egui::RichText::new(format!(
                        "GPUPerfAPI unavailable: {e}\nGPUPerfAPIDX12-x64.dll from the GPA release zip must be on the path (or set AMDGPU_TOP_GPA_DLL)."
                    ))
                    .weak(),
                );
            }
            GpaState::Live { counters, history } => {
                let mut n = 1usize;
                egui::Grid::new("gpa_grid").num_columns(2).show(ui, |ui| {
                    for (i, c) in counters.iter().enumerate() {
                        let pts: Vec<[f64; 2]> = history
                            .iter()
                            .filter_map(|r| r.values.get(i).map(|v| [r.idx, *v]))
                            .collect();
                        sensor_plot(
                            ui,
                            &format!("gpa_{i}"),
                            &c.name,
                            Some(c.mean),
                            "%",
                            Some((0.0, 100.0)),
                            pts,
                        );
                        if n % 2 == 0 {
                            ui.end_row();
                        }
                        n += 1;
                    }
                    if n % 2 == 0 {
                        ui.end_row();
                    }
                });
            }
        }
    });
}

fn activity_section(ui: &mut egui::Ui, dev: &DeviceUi) {
    collapsing(ui, "Activity", true, |ui| {
        let Some(snap) = &dev.last else { return };
        let u = &snap.total_usage;
        let media = u.dec + u.enc + u.media;

        let hist = &dev.engine_history;
        let line = |get: &dyn Fn(&EngineUsage) -> f64| -> Vec<[f64; 2]> {
            hist.iter().map(|r| [r.idx, get(&r.u)]).collect()
        };

        egui::Grid::new("util").num_columns(3).show(ui, |ui| {
            let bar = |ui: &mut egui::Ui, name: &str, v: f64| {
                ui.strong(name);
                ui.add(
                    egui::ProgressBar::new((v.clamp(0.0, 100.0) / 100.0) as f32)
                        .text(format!("{v:.1}%"))
                        .desired_width(140.0),
                );
                ui.end_row();
            };
            bar(ui, "GFX", u.gfx);
            bar(ui, "Media (dec+enc+vpp)", media);
            bar(ui, "Copy", u.dma);
        });

        ui.add_space(4.0);
        plot(
            ui,
            "activity_plot",
            "Engine usage history",
            vec![
                ("GFX %", line(&|u| u.gfx)),
                ("Compute %", line(&|u| u.compute)),
                ("Copy %", line(&|u| u.dma)),
                ("Media %", line(&|u| u.dec + u.enc + u.media)),
            ],
        );
    });
}

fn processes_section(ui: &mut egui::Ui, dev: &DeviceUi, snap: &AdapterSnapshot) {
    collapsing(ui, "Processes", true, |ui| {
        // per-engine totals plot (Linux GUI's fdinfo plot equivalent)
        let hist = &dev.engine_history;
        let line = |get: &dyn Fn(&EngineUsage) -> f64| -> Vec<[f64; 2]> {
            hist.iter().map(|r| [r.idx, get(&r.u)]).collect()
        };
        plot(
            ui,
            "engine_plot",
            "Engine usage history",
            vec![
                ("GFX %", line(&|u| u.gfx)),
                ("Compute %", line(&|u| u.compute)),
                ("Copy %", line(&|u| u.dma)),
                ("Decode %", line(&|u| u.dec)),
                ("Encode %", line(&|u| u.enc)),
                ("VideoProc %", line(&|u| u.media)),
            ],
        );
        ui.add_space(6.0);

        let rows: Vec<_> = snap
            .processes
            .iter()
            .filter(|p| p.usage.total() > 0.05 || p.vram_resident_bytes > 10 * 1048576)
            .collect();

        if rows.is_empty() {
            ui.label("(no active processes)");
            return;
        }

        // Linux GUI layout (egui_grid_fdinfo): short lists expand fully,
        // long lists get a two-axis scroll area with FDINFO_LIST_HEIGHT.
        let proc_len = rows.len();
        let show_procs = |ui: &mut egui::Ui| {
            egui::Grid::new("procs")
                .num_columns(11)
                .striped(true)
                .show(ui, |ui| {
                    for h in [
                        "PID",
                        "NAME",
                        "GFX%",
                        "COMP%",
                        "COPY%",
                        "DEC%",
                        "ENC%",
                        "VPP%",
                        "VRAM MiB",
                        "COMMIT MiB",
                        "SHR MiB",
                    ] {
                        ui.strong(h);
                    }
                    ui.end_row();

                    for p in rows {
                        let u = &p.usage;
                        ui.monospace(p.pid.to_string());
                        ui.label(&p.name);
                        for v in [u.gfx, u.compute, u.dma, u.dec, u.enc, u.media] {
                            ui.monospace(format!("{v:.1}"));
                        }
                        ui.monospace(format!("{:.1}", p.vram_resident_bytes as f64 / 1048576.0));
                        ui.monospace(format!("{:.1}", p.vram_commit_bytes as f64 / 1048576.0));
                        ui.monospace(format!("{:.1}", p.shared_resident_bytes as f64 / 1048576.0));
                        ui.end_row();
                    }
                });
        };

        if proc_len < 8 {
            egui::ScrollArea::horizontal()
                .auto_shrink([false, false])
                .show(ui, show_procs);
        } else {
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .min_scrolled_height(FDINFO_LIST_HEIGHT)
                .show(ui, show_procs);
        }
    });
}

fn vram_section(ui: &mut egui::Ui, dev: &DeviceUi, snap: &AdapterSnapshot) {
    collapsing(ui, "VRAM", true, |ui| {
        let total = (snap.vram_total_kib as f64) * 1024.0;

        // usage bars on top; the bar itself spans 30% of the available width.
        // Bars use adapter-level (device-global) accounting — the same figures as
        // Task Manager's "Dedicated GPU memory" and ADLX GPUVRAM. Per-process sums
        // (bottom text line) only cover user-attributable residency and are
        // always lower: kernel/driver/display-pipeline residency is not attributed
        // to any process (same gap as Σ fdinfo-resident vs vram used on Linux).
        let bar_width = ui.available_width() * 0.3;
        let bar = |ui: &mut egui::Ui, label: &str, used: f64, total: f64| {
            ui.label(label);
            ui.add(
                egui::ProgressBar::new(if total > 0.0 {
                    (used / total).clamp(0.0, 1.0)
                } else {
                    0.0
                } as f32)
                .text(format!(
                    "{:5.0} / {:5.0} MiB",
                    used / 1048576.0,
                    total / 1048576.0
                ))
                .desired_width(bar_width),
            );
        };

        bar(
            ui,
            "Dedicated (device commit — Task Manager / ADLX semantics):",
            snap.vram_commit_device_bytes as f64,
            total,
        );
        bar(
            ui,
            "Dedicated (device resident — physical footprint):",
            snap.vram_resident_device_bytes as f64,
            total,
        );

        let s_total = (snap.shared_total_kib as f64) * 1024.0;
        bar(
            ui,
            "Shared (device resident non-local):",
            snap.shared_resident_device_bytes as f64,
            s_total,
        );

        ui.label(
            egui::RichText::new(format!(
                "Σ per-process: resident {} MiB · commit {} MiB (user-attributable subset)",
                snap.vram_resident_used_bytes / 1048576,
                snap.vram_commit_used_bytes / 1048576,
            ))
            .small()
            .weak(),
        );

        // collapsible history plot below the bars
        ui.add_space(4.0);
        let hist = &dev.vram_history;
        let max_mib = hist
            .iter()
            .map(|r| r.vram_total.max(r.shared_total) as f64 / 1048576.0)
            .last()
            .unwrap_or(0.0);
        let line = |get: &dyn Fn(&VramRecord) -> f64| -> Vec<[f64; 2]> {
            hist.iter().map(|r| [r.idx, get(r)]).collect()
        };
        let mib = |v: u64| v as f64 / 1048576.0;
        collapsing(ui, "VRAM history", true, |ui| {
            Plot::new("vram_plot")
                .allow_scroll(false)
                .include_y(max_mib)
                .height(PLOT_HEIGHT)
                .width(PLOT_WIDTH.min(ui.available_width()))
                .legend(Legend::default())
                .custom_x_axes(vec![time_x_axis()])
                .label_formatter(|name, v| format!("{:.1}s : {name} {:.0} MiB", v.x, v.y))
                .show(ui, |p| {
                    p.line(Line::new(
                        "VRAM commit (device)",
                        line(&|r| mib(r.device_commit)),
                    ));
                    p.line(Line::new(
                        "VRAM resident (device)",
                        line(&|r| mib(r.device_resident)),
                    ));
                    p.line(Line::new(
                        "VRAM resident (Σ proc)",
                        line(&|r| mib(r.resident)),
                    ));
                    p.line(Line::new(
                        "GTT shared (device)",
                        line(&|r| mib(r.device_shared)),
                    ));
                });
        });
    });
}

// ---------------------------------------------------------------------------
// plot widget
// ---------------------------------------------------------------------------

/// Multi-line percent plot (activity / per-engine fdinfo usage), collapsible
/// via its header: y pinned to 0-100 like the Linux GUI's percent plots, x in
/// seconds with a time tooltip.
#[cfg_attr(not(feature = "adlx"), allow(dead_code))]
fn plot(ui: &mut egui::Ui, id: &str, title: &str, lines: Vec<(&str, Vec<[f64; 2]>)>) {
    collapsing(ui, title, true, |ui| {
        Plot::new(id)
            .legend(Legend::default())
            .height(PLOT_HEIGHT)
            .width(PLOT_WIDTH.min(ui.available_width()))
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .include_y(0.0)
            .include_y(100.0)
            .auto_bounds([false, false])
            .custom_x_axes(vec![time_x_axis()])
            .label_formatter(|name, v| format!("{:.1}s : {name} {:.1}%", v.x, v.y))
            .show(ui, |pui| {
                for (name, pts) in lines {
                    pui.line(Line::new(name, pts));
                }
            });
    });
}

fn main() -> eframe::Result<()> {
    let interval = Arc::new(AtomicU64::new(1000));

    // worker thread owns the Sampler (PDH/ADLX handles are not Send);
    // tx moves into the thread, rx into the App
    let (tx, rx) = std::sync::mpsc::channel::<Msg>();
    #[cfg(feature = "gpa")]
    let tx_gpa = tx.clone();
    let interval2 = Arc::clone(&interval);
    std::thread::spawn(move || {
        let mut sampler = match Sampler::new() {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(Msg::Fatal(format!("DXGI/PDH init failed: {e}")));
                return;
            }
        };
        sampler.prime();
        let args = SnapshotArgs { filter_pid: None };
        loop {
            let ms = interval2.load(Ordering::Relaxed).max(50);
            std::thread::sleep(Duration::from_millis(ms));
            let snap = sampler.snapshot(Duration::from_millis(ms), &args);
            if tx.send(Msg::Snap(snap)).is_err() {
                break; // UI closed
            }
        }
    });

    // GPUPerfAPI hardware counters: separate thread because the sampler is
    // !Send AND each capture blocks for seconds (multi-pass + Clear workload).
    #[cfg(feature = "gpa")]
    {
        std::thread::spawn(move || {
            use libamdgpu_top_windows::gpa::{SpmSampler, BUSY_COUNTERS};

            let names: Vec<String> = BUSY_COUNTERS.iter().map(|s| s.to_string()).collect();
            let adapter = match libamdgpu_top_windows::dxgi_first_amd_adapter_handle() {
                Ok(a) => a,
                Err(e) => {
                    let _ = tx_gpa.send(Msg::GpaInitFailed(format!("no AMD adapter: {e}")));
                    return;
                }
            };
            let mut sampler = match SpmSampler::new(adapter, 4096, &names) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[gpa-gui] init failed: {e}");
                    let _ = tx_gpa.send(Msg::GpaInitFailed(e));
                    return;
                }
            };
            eprintln!("[gpa-gui] sampler ready, collecting...");
            loop {
                match sampler.collect() {
                    Ok(counters) if !counters.is_empty() => {
                        if tx_gpa.send(Msg::GpaCounters(counters)).is_err() {
                            break; // UI closed
                        }
                    }
                    // Empty results (GPU idle window): fall through and retry.
                    Ok(_) => {}
                    Err(e) => {
                        let _ = tx_gpa.send(Msg::GpaInitFailed(format!("collect failed: {e}")));
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        });
    }

    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 860.0]),
        ..Default::default()
    };

    eframe::run_native(
        "amdgpu_top_win",
        opts,
        Box::new(move |_cc| Ok(Box::new(App::new(rx, interval)))),
    )
}
