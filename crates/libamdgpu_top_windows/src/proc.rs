//! pid -> process name via Toolhelp32 snapshot (equivalent of parsing /proc/<pid>/comm).

use std::collections::HashMap;
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
};

pub fn get_process_names() -> HashMap<u32, String> {
    let mut out = HashMap::new();

    let Ok(snap) = (unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) }) else {
        return out;
    };

    let mut entry = PROCESSENTRY32W {
        dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
        ..Default::default()
    };

    if unsafe { Process32FirstW(snap, &mut entry) }.is_ok() {
        loop {
            let len = entry
                .szExeFile
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(entry.szExeFile.len());
            out.insert(
                entry.th32ProcessID,
                String::from_utf16_lossy(&entry.szExeFile[..len]),
            );
            if unsafe { Process32NextW(snap, &mut entry) }.is_err() {
                break;
            }
        }
    }

    let _ = unsafe { windows::Win32::Foundation::CloseHandle(snap) };
    out
}
