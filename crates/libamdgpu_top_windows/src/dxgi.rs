//! DXGI adapter enumeration (equivalent of amdgpu_top's `DevicePath::get_device_path_list`).

use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
};

pub struct WinGpuAdapter {
    pub description: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub luid_high: i32,
    pub luid_low: u32,
    /// bytes, ≈ VRAM total
    pub dedicated_video_memory: u64,
    /// bytes
    pub dedicated_system_memory: u64,
    /// bytes, ≈ GTT/shared total
    pub shared_system_memory: u64,
    /// DXGI_ADAPTER_DESC::SubSysId
    pub sub_sys_id: u32,
    /// DXGI_ADAPTER_DESC::Revision (PCI revision id)
    pub revision: u32,
}

pub const AMD_VENDOR_ID: u32 = 0x1002;

/// Enumerate AMD (vendor 0x1002) hardware adapters. Software adapters are skipped.
pub fn enumerate_amd_adapters() -> windows::core::Result<Vec<WinGpuAdapter>> {
    enumerate_by_vendor(AMD_VENDOR_ID)
}

/// First AMD hardware adapter handle (for D3D12 device creation / GPA).
#[cfg_attr(not(feature = "gpa"), allow(dead_code))]
pub fn first_amd_adapter_handle() -> windows::core::Result<IDXGIAdapter1> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }?;
    for index in 0.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
            break;
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if desc.VendorId == AMD_VENDOR_ID && desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            return Ok(adapter);
        }
    }
    Err(windows::core::Error::from_win32())
}

pub fn enumerate_by_vendor(vendor_id: u32) -> windows::core::Result<Vec<WinGpuAdapter>> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }?;
    let mut out = Vec::new();

    for index in 0.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
            break;
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };

        if desc.VendorId != vendor_id {
            continue;
        }
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }

        let len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        out.push(WinGpuAdapter {
            description: String::from_utf16_lossy(&desc.Description[..len]),
            vendor_id: desc.VendorId,
            device_id: desc.DeviceId,
            luid_high: desc.AdapterLuid.HighPart,
            luid_low: desc.AdapterLuid.LowPart,
            dedicated_video_memory: desc.DedicatedVideoMemory as u64,
            dedicated_system_memory: desc.DedicatedSystemMemory as u64,
            shared_system_memory: desc.SharedSystemMemory as u64,
            sub_sys_id: unsafe { adapter.GetDesc() }.map(|d| d.SubSysId).unwrap_or(0),
            revision: unsafe { adapter.GetDesc() }.map(|d| d.Revision).unwrap_or(0),
        });
    }

    Ok(out)
}
