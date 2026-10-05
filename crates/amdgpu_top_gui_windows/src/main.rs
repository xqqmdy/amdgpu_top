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
use egui_plot::{Legend, Line, Plot};
use libamdgpu_top_windows::{AdapterSnapshot, Sampler, SnapshotArgs};
#[cfg(any(feature = "adlx", feature = "gpa"))]
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "adlx")]
use libamdgpu_top_windows::SensorSnapshot;

const MAX_SAMPLES: usize = 240;

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

/// VRAM history sample (always collected, like the Linux GUI's vram_history)
struct VramRecord {
    idx: f64,
    /// Σ per-process resident local bytes
    resident: u64,
    /// Σ per-process committed dedicated bytes
    commit: u64,
    /// Σ per-process resident non-local bytes
    shared: u64,
    /// adapter totals, bytes
    vram_total: u64,
    shared_total: u64,
}

struct DeviceUi {
    description: String,
    device_id: u32,
    luid: String,
    last: Option<AdapterSnapshot>,
    sample_idx: f64,
    vram_history: std::collections::VecDeque<VramRecord>,
    #[cfg(feature = "adlx")]
    history: VecDeque<SampleRecord>,
}

impl DeviceUi {
    fn push_vram(&mut self, dev: &AdapterSnapshot) {
        self.vram_history.push_back(VramRecord {
            idx: self.sample_idx,
            resident: dev.vram_resident_used_bytes,
            commit: dev.vram_commit_used_bytes,
            shared: dev.shared_resident_used_bytes,
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

#[cfg(feature = "gpa")]
struct CounterRecord {
    idx: f64,
    gpu: f64,
    cs: f64,
    tex: f64,
    mem: f64,
}

#[cfg(feature = "gpa")]
impl CounterRecord {
    fn from_counters(idx: f64, counters: &[libamdgpu_top_windows::gpa::SpmCounter]) -> Self {
        let get = |name: &str| {
            counters
                .iter()
                .find(|c| c.name == name)
                .map(|c| c.mean)
                .unwrap_or(f64::NAN)
        };
        Self {
            idx,
            gpu: get("GPUBusy"),
            cs: get("CSBusy"),
            tex: get("TexUnitBusy"),
            mem: get("MemUnitBusy"),
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
                    let idx = history.back().map(|r| r.idx + 1.0).unwrap_or(1.0);
                    history.push_back(CounterRecord::from_counters(idx, &counters));
                    while history.len() > MAX_SAMPLES {
                        history.pop_front();
                    }
                    self.gpa = GpaState::Live { counters, history };
                }
                Msg::Snap(snap) => {
                    for (i, dev) in snap.iter().enumerate() {
                        if let Some(slot) = self.devices.get_mut(i) {
                            slot.description = dev.description.clone();
                            slot.device_id = dev.device_id;
                            slot.luid = dev.luid.clone();
                            slot.sample_idx += 1.0;
                            slot.push_vram(dev);
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
                                luid: dev.luid.clone(),
                                last: Some(dev.clone()),
                                sample_idx: 1.0,
                                vram_history: std::collections::VecDeque::new(),
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

#[cfg(feature = "adlx")]
fn series(
    history: &VecDeque<SampleRecord>,
    get: impl Fn(&SensorSnapshot) -> Option<f64>,
) -> Vec<[f64; 2]> {
    history
        .iter()
        .filter_map(|r| get(r.s.as_ref()?).map(|v| [r.idx, v]))
        .collect()
}

#[cfg_attr(not(feature = "adlx"), allow(dead_code))]
fn fmt_opt(v: Option<f64>, unit: &str) -> String {
    match v {
        Some(v) => format!("{v:.1}{unit}"),
        None => "-".into(),
    }
}

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

            // single page, all sections stacked vertically (like the Linux GUI)
            egui::ScrollArea::vertical().show(ui, |ui| {
                sensors_section(ui, dev, snap);
                ui.add_space(8.0);
                utilization_section(ui, snap);
                ui.add_space(8.0);
                #[cfg(feature = "gpa")]
                hardware_counters_section(ui, &self.gpa);
                #[cfg(feature = "gpa")]
                ui.add_space(8.0);
                vram_section(ui, dev, snap);
                ui.add_space(8.0);
                processes_section(ui, snap);
            });
        });
    }
}

// ---------------------------------------------------------------------------
// sections (single page)
// ---------------------------------------------------------------------------

#[cfg_attr(not(feature = "adlx"), allow(unused_variables))]
fn sensors_section(ui: &mut egui::Ui, dev: &DeviceUi, snap: &AdapterSnapshot) {
    ui.heading("Sensors");

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
        match &snap.sensors {
            Some(s) => {
                egui::Grid::new("sensor_now").num_columns(4).show(ui, |ui| {
                    let row = |ui: &mut egui::Ui, k: &str, v: Option<f64>, unit: &str| {
                        ui.strong(k);
                        ui.monospace(fmt_opt(v, unit));
                    };
                    row(ui, "GPU usage", s.gpu_usage, " %");
                    row(ui, "GPU clock", s.gpu_clock_mhz, " MHz");
                    row(ui, "VRAM clock", s.vram_clock_mhz, " MHz");
                    ui.end_row();
                    row(ui, "Temp (edge)", s.temp_edge_c, " °C");
                    row(ui, "Temp (hotspot)", s.temp_hotspot_c, " °C");
                    row(ui, "Fan", s.fan_rpm, " RPM");
                    ui.end_row();
                    row(ui, "Board power", s.total_board_power_w, " W");
                    row(ui, "Chip power", s.power_w, " W");
                    row(ui, "Voltage", s.voltage_mv, " mV");
                    ui.end_row();
                    row(ui, "VRAM used (ADLX)", s.vram_used_mb, " MB");
                    ui.label("");
                    ui.label("");
                    ui.end_row();
                });

                ui.add_space(4.0);
                let hist = &dev.history;
                plot(
                    ui,
                    "p_temp",
                    vec![
                        ("edge °C", series(hist, |s| s.temp_edge_c)),
                        ("hotspot °C", series(hist, |s| s.temp_hotspot_c)),
                    ],
                );
                plot(
                    ui,
                    "p_clk",
                    vec![
                        ("GPU MHz", series(hist, |s| s.gpu_clock_mhz)),
                        ("VRAM MHz", series(hist, |s| s.vram_clock_mhz)),
                    ],
                );
                plot(
                    ui,
                    "p_pow",
                    vec![
                        ("board W", series(hist, |s| s.total_board_power_w)),
                        ("fan ÷10 RPM", series(hist, |s| s.fan_rpm.map(|v| v / 10.0))),
                    ],
                );
            }
            None => {
                ui.label(
                    egui::RichText::new(
                        "ADLX sensors unavailable for this adapter (shadow adapter or driver without ADLX).",
                    )
                    .weak(),
                );
            }
        }
    }
}

/// Hardware IP busy counters via GPUPerfAPI (first AMD adapter).
/// Values come from a slow blocking sampler thread (one capture takes
/// seconds: multi-pass + built-in Clear workload), so they refresh at
/// their own pace, independent of the PDH snapshot interval.
#[cfg(feature = "gpa")]
fn hardware_counters_section(ui: &mut egui::Ui, gpa: &GpaState) {
    ui.heading("Hardware Counters (GPUPerfAPI)");
    ui.label(
        egui::RichText::new(
            "device-global HW counters, first AMD adapter · window = built-in workload (self-load inflates GPU-busy-class counters)",
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
            egui::Grid::new("gpa_counters")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    for c in counters {
                        ui.strong(&c.name);
                        ui.add(
                            egui::ProgressBar::new((c.mean.clamp(0.0, 100.0) / 100.0) as f32)
                                .text(format!("{:.2}%", c.mean))
                                .desired_width(200.0),
                        );
                        ui.end_row();
                    }
                });

            ui.add_space(4.0);
            let line = |get: &dyn Fn(&CounterRecord) -> f64| -> Vec<[f64; 2]> {
                history.iter().map(|r| [r.idx, get(r)]).collect()
            };
            plot(
                ui,
                "gpa_busy",
                vec![
                    ("GPUBusy %", line(&|r| r.gpu)),
                    ("CSBusy %", line(&|r| r.cs)),
                    ("TexUnitBusy %", line(&|r| r.tex)),
                    ("MemUnitBusy %", line(&|r| r.mem)),
                ],
            );
        }
    }
}

fn utilization_section(ui: &mut egui::Ui, snap: &AdapterSnapshot) {
    ui.heading("Utilization");
    let u = &snap.total_usage;

    egui::Grid::new("util").num_columns(6).show(ui, |ui| {
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
        bar(ui, "Compute", u.compute);
        bar(ui, "Copy", u.dma);
        bar(ui, "Decode", u.dec);
        bar(ui, "Encode", u.enc);
        bar(ui, "VideoProc", u.media);
    });
}

fn processes_section(ui: &mut egui::Ui, snap: &AdapterSnapshot) {
    ui.heading("Processes");
    let rows: Vec<_> = snap
        .processes
        .iter()
        .filter(|p| p.usage.total() > 0.05 || p.vram_resident_bytes > 10 * 1048576)
        .collect();

    if rows.is_empty() {
        ui.label("(no active processes)");
        return;
    }

    egui::ScrollArea::vertical()
        .max_height(280.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
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
        });
}

fn vram_section(ui: &mut egui::Ui, dev: &DeviceUi, snap: &AdapterSnapshot) {
    ui.heading("VRAM");
    let total = (snap.vram_total_kib as f64) * 1024.0;

    // usage bars on top; the bar itself spans 60% of the available width
    let bar_width = ui.available_width() * 0.6;
    let bar = |ui: &mut egui::Ui, label: &str, used: f64| {
        ui.label(label);
        ui.add(
            egui::ProgressBar::new(if total > 0.0 {
                (used / total).clamp(0.0, 1.0)
            } else {
                0.0
            } as f32)
            .text(format!(
                "{:.2} GiB / {:.2} GiB",
                used / 1073741824.0,
                total / 1073741824.0
            ))
            .desired_width(bar_width),
        );
    };

    bar(
        ui,
        "Resident (Σ per-process — real footprint, replay buffers excluded):",
        snap.vram_resident_used_bytes as f64,
    );
    bar(
        ui,
        "Committed (includes evictable driver buffers, e.g. ReLive replay):",
        snap.vram_commit_used_bytes as f64,
    );

    let s_total = (snap.shared_total_kib as f64) * 1024.0;
    let s_used = snap.shared_resident_used_bytes as f64;
    ui.label("Shared GPU memory (resident non-local):");
    ui.add(
        egui::ProgressBar::new(if s_total > 0.0 {
            (s_used / s_total).clamp(0.0, 1.0)
        } else {
            0.0
        } as f32)
        .text(format!(
            "{:.2} GiB / {:.2} GiB",
            s_used / 1073741824.0,
            s_total / 1073741824.0
        ))
        .desired_width(bar_width),
    );

    // history plot below, always expanded
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
    Plot::new("vram_plot")
        .allow_scroll(false)
        .include_y(max_mib)
        .height(110.0)
        .width(ui.available_width())
        .legend(Legend::default())
        .show(ui, |p| {
            p.line(Line::new(
                "VRAM (resident)",
                line(&|r| r.resident as f64 / 1048576.0),
            ));
            p.line(Line::new(
                "VRAM (commit)",
                line(&|r| r.commit as f64 / 1048576.0),
            ));
            p.line(Line::new(
                "GTT (shared)",
                line(&|r| r.shared as f64 / 1048576.0),
            ));
        });
}

// ---------------------------------------------------------------------------
// plot widget
// ---------------------------------------------------------------------------

#[cfg_attr(not(feature = "adlx"), allow(dead_code))]
fn plot(ui: &mut egui::Ui, id: &str, lines: Vec<(&str, Vec<[f64; 2]>)>) {
    Plot::new(id)
        .legend(Legend::default())
        .height(110.0)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .show(ui, |pui| {
            for (name, pts) in lines {
                pui.line(Line::new(name, pts));
            }
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
                // ~200 ms/pass sampling window (TDR-safe; see SpmSampler::collect)
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
