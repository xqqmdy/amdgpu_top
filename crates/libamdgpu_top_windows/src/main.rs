//! amdgpu_top_win: Windows port of amdgpu_top (P0 PoC).
//!
//! Usage:
//!   amdgpu_top_win                  interactive table (refresh 1000 ms)
//!   amdgpu_top_win -J               NDJSON stream, one object per update
//!   amdgpu_top_win -J --once        single JSON object then exit
//!   amdgpu_top_win -u 500           update interval in milliseconds
//!   amdgpu_top_win --pid 1234       filter one process

#![cfg(windows)]

use libamdgpu_top_windows::{AdapterSnapshot, ProcGpuUsage, Sampler, SnapshotArgs};
use std::io::{self, Write};
use std::time::{Duration, Instant};

const CSI_CLEAR: &str = "\x1b[2J\x1b[H";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut json = false;
    let mut once = false;
    let mut show_all = false;
    let mut interval_ms: u64 = 1000;
    let mut filter_pid: Option<u32> = None;
    #[cfg(feature = "gpa")]
    let mut spm = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-J" | "--json" => json = true,
            "--once" => once = true,
            "-a" | "--all" => show_all = true,
            #[cfg(feature = "gpa")]
            "--spm" => spm = true,
            "-u" | "--update-ms" => {
                interval_ms = it.next().and_then(|v| v.parse().ok()).unwrap_or(1000)
            }
            "-p" | "--pid" => filter_pid = it.next().and_then(|v| v.parse().ok()),
            "-h" | "--help" => {
                println!(
                    "amdgpu_top_win [-J|--json] [--once] [-a|--all] [-u <ms>] [-p <pid>]{}",
                    if cfg!(feature = "gpa") {
                        " [--spm]"
                    } else {
                        ""
                    }
                );
                return;
            }
            _ => {}
        }
    }

    let interval = Duration::from_millis(interval_ms.max(50));
    let mut sampler = match Sampler::new() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to init DXGI/PDH: {e}");
            std::process::exit(1);
        }
    };

    let snap_args = SnapshotArgs { filter_pid };

    #[cfg(feature = "gpa")]
    if spm {
        return spm_main(interval_ms);
    }

    // Prime PDH rate counters (utilization needs 2 samples spaced in time).
    sampler.prime();
    std::thread::sleep(interval);

    loop {
        let start = Instant::now();
        let snapshot = sampler.snapshot(interval, &snap_args);

        if json {
            let line = serde_json::to_string(&snapshot).unwrap_or_else(|_| "[]".into());
            let mut out = io::stdout().lock();
            // tolerate closed pipes (e.g. `| head`)
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        } else {
            let mut out = io::stdout().lock();
            let _ = write!(out, "{CSI_CLEAR}");
            let _ = render_table(&mut out, &snapshot, interval_ms, show_all);
            let _ = out.flush();
        }

        if once {
            break;
        }
        let elapsed = start.elapsed();
        if elapsed < interval {
            std::thread::sleep(interval - elapsed);
        }
    }
}

fn mib_bytes(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / 1048576.0)
}

fn mib(kib: u64) -> String {
    format!("{:.1}", kib as f64 / 1024.0)
}

fn render_table(
    out: &mut impl Write,
    snapshot: &[AdapterSnapshot],
    interval_ms: u64,
    show_all: bool,
) -> io::Result<()> {
    if snapshot.is_empty() {
        return writeln!(out, "No AMD GPU (vendor 0x1002) found via DXGI.");
    }

    writeln!(
        out,
        "amdgpu_top_win (DXGI + PDH backend)   update: {interval_ms} ms"
    )?;
    writeln!(out)?;

    for dev in snapshot {
        writeln!(
            out,
            "{}  [device id 0x{:04X}, luid {}]",
            dev.description, dev.device_id, dev.luid
        )?;
        writeln!(
            out,
            "  VRAM total {} MiB | resident(Σproc) {} MiB | commit(Σproc) {} MiB",
            mib(dev.vram_total_kib),
            mib_bytes(dev.vram_resident_used_bytes),
            mib_bytes(dev.vram_commit_used_bytes),
        )?;
        let t = &dev.total_usage;
        writeln!(
            out,
            "  Total  GFX {:5.1}%  COMP {:5.1}%  COPY {:5.1}%  DEC {:5.1}%  ENC {:5.1}%  VPP {:5.1}%",
            t.gfx, t.compute, t.dma, t.dec, t.enc, t.media
        )?;

        #[cfg(feature = "adlx")]
        if let Some(s) = &dev.sensors {
            let opt = |v: Option<f64>| match v {
                Some(v) => format!("{v:>6.1}"),
                None => "     -".to_string(),
            };
            writeln!(
                out,
                "  SNSR  usage {}%  clk {} MHz  memclk {} MHz  temp {} C  hot {} C  fan {} RPM  board {} W  volt {} mV  vram {} MB",
                opt(s.gpu_usage),
                opt(s.gpu_clock_mhz),
                opt(s.vram_clock_mhz),
                opt(s.temp_edge_c),
                opt(s.temp_hotspot_c),
                opt(s.fan_rpm),
                opt(s.total_board_power_w),
                opt(s.voltage_mv),
                opt(s.vram_used_mb),
            )?;
        }

        let rows: Vec<&ProcGpuUsage> = dev
            .processes
            .iter()
            .filter(|p| show_all || p.usage.total() > 0.05 || p.vram_resident_bytes > 10 * 1048576)
            .collect();

        if rows.is_empty() {
            writeln!(out, "  (no active processes; use -a to show all)")?;
            continue;
        }

        writeln!(
            out,
            "  {:>7}  {:<24} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10}",
            "PID",
            "NAME",
            "GFX%",
            "COMP%",
            "COPY%",
            "DEC%",
            "ENC%",
            "VPP%",
            "VRAM(MiB)",
            "COMMIT(MiB)",
            "SHR(MiB)"
        )?;
        for p in rows {
            let u = &p.usage;
            writeln!(
                out,
                "  {:>7}  {:<24} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>10} {:>10} {:>10}",
                p.pid,
                truncate(&p.name, 24),
                u.gfx, u.compute, u.dma, u.dec, u.enc, u.media,
                mib_bytes(p.vram_resident_bytes),
                mib_bytes(p.vram_commit_bytes),
                mib_bytes(p.shared_resident_bytes),
            )?;
        }
        writeln!(out)?;
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max - 1).collect::<String>() + "…"
    }
}

/// SPM mode: hardware counter sampling via GPUPerfAPI (GRBM-class IP busy).
#[cfg(feature = "gpa")]
fn spm_main(interval_ms: u64) {
    use libamdgpu_top_windows::gpa::{SpmSampler, BUSY_COUNTERS};

    let names: Vec<String> = BUSY_COUNTERS.iter().map(|s| s.to_string()).collect();

    let mut sampler = match SpmSampler::new(
        libamdgpu_top_windows::dxgi_first_amd_adapter_handle().expect("no AMD adapter"),
        4096,
        &names,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SPM init failed: {e}");
            std::process::exit(1);
        }
    };

    let preview = sampler.list_counters(12);
    eprintln!("[gpa] first public counters: {preview:?}");

    // Warm-up round (driver lazily creates SPM resources on first use).
    let _ = sampler.collect();

    loop {
        let start = Instant::now();
        match sampler.collect() {
            Ok(counters) if !counters.is_empty() => {
                println!("{CSI_CLEAR}");
                println!("SPM hardware counters (GPUPerfAPI)  zero-workload ~500 ms/pass window");
                println!();
                println!(
                    "  {:<26} {:>12} {:>12} {:>8}",
                    "COUNTER", "MEAN", "LAST", "SAMPLES"
                );
                for c in &counters {
                    println!(
                        "  {:<26} {:>12.2} {:>12.2} {:>8}",
                        c.name, c.mean, c.last, c.samples
                    );
                }
            }
            Ok(_) => eprintln!("[gpa] no SPM data (GPU idle? try a GPU workload)"),
            Err(e) => {
                eprintln!("SPM collect failed: {e}");
                std::process::exit(1);
            }
        }
        let elapsed = start.elapsed();
        let interval = Duration::from_millis(interval_ms.max(50));
        if elapsed < interval {
            std::thread::sleep(interval - elapsed);
        }
    }
}
