//! cuGetProcAddress implementation — the dispatch table for all CUDA functions.
//!
//! This is CRITICAL for PyTorch compatibility. PyTorch looks up all CUDA functions
//! at runtime via cuGetProcAddress rather than linking directly.

use std::ffi::{c_char, c_int, c_void, CStr};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
const CUDA_ERROR_NOT_FOUND: CUresult = 500;

// Import all exported functions from our modules
use crate::error::{cuGetErrorName, cuGetErrorString};
use crate::stubs;

/// Look up a CUDA driver API function by name and return its function pointer.
///
/// This is the main dispatch table. Every exported CUDA function must be listed here.
/// PyTorch and other CUDA runtimes use this to discover available functions.
#[no_mangle]
pub unsafe extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    _cuda_version: c_int,
    _flags: u64,
    symbol_status: *mut c_int,
) -> CUresult {
    if symbol.is_null() || pfn.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let name = match CStr::from_ptr(symbol).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !symbol_status.is_null() {
                *symbol_status = 2; // not found
            }
            return CUDA_ERROR_NOT_FOUND;
        }
    };

    let func_ptr: Option<*mut c_void> = match name {
        // ── Initialization ──────────────────────────────────────
        "cuInit" => Some(crate::cuInit as *mut c_void),
        "cuDriverGetVersion" => Some(crate::cuDriverGetVersion as *mut c_void),

        // ── Error Handling ──────────────────────────────────────
        "cuGetErrorString" => Some(cuGetErrorString as *mut c_void),
        "cuGetErrorName" => Some(cuGetErrorName as *mut c_void),

        // ── Device Management ───────────────────────────────────
        "cuDeviceGetCount" => Some(crate::cuDeviceGetCount as *mut c_void),
        "cuDeviceGet" => Some(crate::cuDeviceGet as *mut c_void),
        "cuDeviceGetName" => Some(crate::cuDeviceGetName as *mut c_void),
        "cuDeviceGetAttribute" => Some(crate::cuDeviceGetAttribute as *mut c_void),
        "cuDeviceTotalMem" | "cuDeviceTotalMem_v2" => {
            Some(crate::cuDeviceTotalMem_v2 as *mut c_void)
        }
        "cuDeviceComputeCapability" => Some(crate::cuDeviceComputeCapability as *mut c_void),
        "cuDeviceGetUuid" | "cuDeviceGetUuid_v2" => Some(crate::cuDeviceGetUuid as *mut c_void),
        "cuDeviceGetP2PAttribute" => Some(crate::cuDeviceGetP2PAttribute as *mut c_void),
        "cuDeviceCanAccessPeer" => Some(crate::cuDeviceCanAccessPeer as *mut c_void),
        "cuDeviceGetByPCIBusId" => Some(crate::cuDeviceGetByPCIBusId as *mut c_void),
        "cuDeviceGetPCIBusId" => Some(crate::cuDeviceGetPCIBusId as *mut c_void),
        "cuDeviceGetDefaultMemPool" => Some(crate::cuDeviceGetDefaultMemPool as *mut c_void),
        "cuDeviceGetMemPool" => Some(crate::cuDeviceGetMemPool as *mut c_void),
        "cuDeviceSetMemPool" => Some(crate::cuDeviceSetMemPool as *mut c_void),

        // ── Primary Context ─────────────────────────────────────
        "cuDevicePrimaryCtxRetain" => Some(crate::cuDevicePrimaryCtxRetain as *mut c_void),
        "cuDevicePrimaryCtxRelease" | "cuDevicePrimaryCtxRelease_v2" => {
            Some(crate::cuDevicePrimaryCtxRelease_v2 as *mut c_void)
        }
        "cuDevicePrimaryCtxReset" | "cuDevicePrimaryCtxReset_v2" => {
            Some(crate::cuDevicePrimaryCtxReset_v2 as *mut c_void)
        }
        "cuDevicePrimaryCtxGetState" => Some(crate::cuDevicePrimaryCtxGetState as *mut c_void),
        "cuDevicePrimaryCtxSetFlags" | "cuDevicePrimaryCtxSetFlags_v2" => {
            Some(crate::cuDevicePrimaryCtxSetFlags_v2 as *mut c_void)
        }

        // ── Context Management ──────────────────────────────────
        "cuCtxCreate" | "cuCtxCreate_v2" => Some(crate::cuCtxCreate_v2 as *mut c_void),
        "cuCtxDestroy" | "cuCtxDestroy_v2" => Some(crate::cuCtxDestroy_v2 as *mut c_void),
        "cuCtxSetCurrent" => Some(crate::cuCtxSetCurrent as *mut c_void),
        "cuCtxGetCurrent" => Some(crate::cuCtxGetCurrent as *mut c_void),
        "cuCtxSynchronize" => Some(crate::cuCtxSynchronize as *mut c_void),
        "cuCtxPushCurrent" | "cuCtxPushCurrent_v2" => {
            Some(crate::cuCtxPushCurrent_v2 as *mut c_void)
        }
        "cuCtxPopCurrent" | "cuCtxPopCurrent_v2" => {
            Some(crate::cuCtxPopCurrent_v2 as *mut c_void)
        }
        "cuCtxGetDevice" => Some(crate::cuCtxGetDevice as *mut c_void),
        "cuCtxSetCacheConfig" => Some(crate::cuCtxSetCacheConfig as *mut c_void),
        "cuCtxGetCacheConfig" => Some(crate::cuCtxGetCacheConfig as *mut c_void),
        "cuCtxSetLimit" => Some(crate::cuCtxSetLimit as *mut c_void),
        "cuCtxGetLimit" => Some(crate::cuCtxGetLimit as *mut c_void),
        "cuCtxGetStreamPriorityRange" => Some(crate::cuCtxGetStreamPriorityRange as *mut c_void),
        "cuCtxGetApiVersion" => Some(crate::cuCtxGetApiVersion as *mut c_void),
        "cuCtxGetFlags" => Some(crate::cuCtxGetFlags as *mut c_void),
        "cuCtxSetFlags" => Some(crate::cuCtxSetFlags as *mut c_void),
        "cuCtxResetPersistingL2Cache" => Some(crate::cuCtxResetPersistingL2Cache as *mut c_void),

        // ── Peer Access ─────────────────────────────────────────
        "cuCtxEnablePeerAccess" => Some(crate::cuCtxEnablePeerAccess as *mut c_void),
        "cuCtxDisablePeerAccess" => Some(crate::cuCtxDisablePeerAccess as *mut c_void),

        // ── Module Management ───────────────────────────────────
        "cuModuleLoadData" => Some(crate::cuModuleLoadData as *mut c_void),
        "cuModuleUnload" => Some(crate::cuModuleUnload as *mut c_void),
        "cuModuleGetFunction" => Some(crate::cuModuleGetFunction as *mut c_void),
        "cuModuleLoad" => Some(crate::cuModuleLoad as *mut c_void),
        "cuModuleLoadDataEx" => Some(crate::cuModuleLoadDataEx as *mut c_void),
        "cuModuleLoadFatBinary" => Some(crate::cuModuleLoadFatBinary as *mut c_void),
        "cuModuleGetGlobal" | "cuModuleGetGlobal_v2" => {
            Some(crate::cuModuleGetGlobal_v2 as *mut c_void)
        }

        // ── Linker ──────────────────────────────────────────────
        "cuLinkCreate" | "cuLinkCreate_v2" => Some(crate::cuLinkCreate_v2 as *mut c_void),
        "cuLinkAddData" | "cuLinkAddData_v2" => Some(crate::cuLinkAddData_v2 as *mut c_void),
        "cuLinkAddFile" | "cuLinkAddFile_v2" => Some(crate::cuLinkAddFile_v2 as *mut c_void),
        "cuLinkComplete" => Some(crate::cuLinkComplete as *mut c_void),
        "cuLinkDestroy" => Some(crate::cuLinkDestroy as *mut c_void),

        // ── Memory Management ───────────────────────────────────
        "cuMemAlloc" | "cuMemAlloc_v2" => Some(crate::cuMemAlloc_v2 as *mut c_void),
        "cuMemFree" | "cuMemFree_v2" => Some(crate::cuMemFree_v2 as *mut c_void),
        "cuMemcpyHtoD" | "cuMemcpyHtoD_v2" => Some(crate::cuMemcpyHtoD_v2 as *mut c_void),
        "cuMemcpyDtoH" | "cuMemcpyDtoH_v2" => Some(crate::cuMemcpyDtoH_v2 as *mut c_void),
        "cuMemcpyDtoD" | "cuMemcpyDtoD_v2" => Some(crate::cuMemcpyDtoD_v2 as *mut c_void),
        "cuMemcpyHtoDAsync" | "cuMemcpyHtoDAsync_v2" => {
            Some(crate::cuMemcpyHtoDAsync_v2 as *mut c_void)
        }
        "cuMemcpyDtoHAsync" | "cuMemcpyDtoHAsync_v2" => {
            Some(crate::cuMemcpyDtoHAsync_v2 as *mut c_void)
        }
        "cuMemcpyDtoDAsync" | "cuMemcpyDtoDAsync_v2" => {
            Some(crate::cuMemcpyDtoDAsync_v2 as *mut c_void)
        }
        "cuMemsetD8" | "cuMemsetD8_v2" => Some(crate::cuMemsetD8_v2 as *mut c_void),
        "cuMemsetD16" | "cuMemsetD16_v2" => Some(crate::cuMemsetD16_v2 as *mut c_void),
        "cuMemsetD32" | "cuMemsetD32_v2" => Some(crate::cuMemsetD32_v2 as *mut c_void),
        "cuMemGetInfo" | "cuMemGetInfo_v2" => Some(crate::cuMemGetInfo_v2 as *mut c_void),
        "cuMemGetAddressRange" | "cuMemGetAddressRange_v2" => {
            Some(crate::cuMemGetAddressRange_v2 as *mut c_void)
        }
        "cuMemAllocHost" | "cuMemAllocHost_v2" => Some(crate::cuMemAllocHost_v2 as *mut c_void),
        "cuMemFreeHost" => Some(crate::cuMemFreeHost as *mut c_void),
        "cuMemHostAlloc" => Some(crate::cuMemHostAlloc as *mut c_void),
        "cuMemHostGetDevicePointer" | "cuMemHostGetDevicePointer_v2" => {
            Some(crate::cuMemHostGetDevicePointer_v2 as *mut c_void)
        }
        "cuMemHostGetFlags" => Some(crate::cuMemHostGetFlags as *mut c_void),
        "cuMemAllocManaged" => Some(crate::cuMemAllocManaged as *mut c_void),
        "cuMemAllocPitch" | "cuMemAllocPitch_v2" => {
            Some(crate::cuMemAllocPitch_v2 as *mut c_void)
        }

        // ── Memory Pools ────────────────────────────────────────
        "cuMemPoolDestroy" => Some(crate::cuMemPoolDestroy as *mut c_void),
        "cuMemPoolTrimTo" => Some(crate::cuMemPoolTrimTo as *mut c_void),
        "cuMemAllocAsync" | "cuMemAllocAsync_ptsz" => Some(crate::cuMemAllocAsync as *mut c_void),
        "cuMemFreeAsync" | "cuMemFreeAsync_ptsz" => Some(crate::cuMemFreeAsync as *mut c_void),
        "cuMemAllocFromPoolAsync" | "cuMemAllocFromPoolAsync_ptsz" => {
            Some(crate::cuMemAllocFromPoolAsync as *mut c_void)
        }

        // ── Execution Control ───────────────────────────────────
        "cuLaunchKernel" => Some(crate::cuLaunchKernel as *mut c_void),
        "cuLaunchCooperativeKernel" => Some(crate::cuLaunchCooperativeKernel as *mut c_void),
        "cuFuncGetAttribute" => Some(crate::cuFuncGetAttribute as *mut c_void),
        "cuFuncSetAttribute" => Some(crate::cuFuncSetAttribute as *mut c_void),
        "cuFuncSetCacheConfig" => Some(crate::cuFuncSetCacheConfig as *mut c_void),
        "cuFuncSetSharedMemConfig" => Some(crate::cuFuncSetSharedMemConfig as *mut c_void),
        "cuFuncGetModule" => Some(crate::cuFuncGetModule as *mut c_void),
        "cuFuncGetName" => Some(crate::cuFuncGetName as *mut c_void),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor" => {
            Some(crate::cuOccupancyMaxActiveBlocksPerMultiprocessor as *mut c_void)
        }
        "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" => {
            Some(crate::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags as *mut c_void)
        }
        "cuOccupancyAvailableDynamicSMemPerBlock" => {
            Some(crate::cuOccupancyAvailableDynamicSMemPerBlock as *mut c_void)
        }

        // ── Stream Management ───────────────────────────────────
        "cuStreamCreate" => Some(crate::cuStreamCreate as *mut c_void),
        "cuStreamCreateWithPriority" => Some(crate::cuStreamCreateWithPriority as *mut c_void),
        "cuStreamDestroy" | "cuStreamDestroy_v2" => {
            Some(crate::cuStreamDestroy_v2 as *mut c_void)
        }
        "cuStreamSynchronize" => Some(crate::cuStreamSynchronize as *mut c_void),
        "cuStreamQuery" => Some(crate::cuStreamQuery as *mut c_void),
        "cuStreamWaitEvent" => Some(crate::cuStreamWaitEvent as *mut c_void),
        "cuStreamGetPriority" => Some(crate::cuStreamGetPriority as *mut c_void),
        "cuStreamGetFlags" => Some(crate::cuStreamGetFlags as *mut c_void),
        "cuStreamGetCtx" | "cuStreamGetCtx_v2" => Some(crate::cuStreamGetCtx_v2 as *mut c_void),

        // ── Event Management ────────────────────────────────────
        "cuEventCreate" => Some(crate::cuEventCreate as *mut c_void),
        "cuEventDestroy" | "cuEventDestroy_v2" => Some(crate::cuEventDestroy_v2 as *mut c_void),
        "cuEventRecord" => Some(crate::cuEventRecord as *mut c_void),
        "cuEventRecordWithFlags" => Some(crate::cuEventRecordWithFlags as *mut c_void),
        "cuEventSynchronize" => Some(crate::cuEventSynchronize as *mut c_void),
        "cuEventQuery" => Some(crate::cuEventQuery as *mut c_void),
        "cuEventElapsedTime" => Some(crate::cuEventElapsedTime as *mut c_void),

        // ── Pointer Queries ─────────────────────────────────────
        "cuPointerGetAttribute" => Some(crate::cuPointerGetAttribute as *mut c_void),
        "cuPointerSetAttribute" => Some(crate::cuPointerSetAttribute as *mut c_void),

        // ── Proc Address (self-referential) ─────────────────────
        "cuGetProcAddress" => Some(cuGetProcAddress as *mut c_void),
        "cuGetProcAddress_v2" => Some(cuGetProcAddress_v2 as *mut c_void),

        // ── Stubs (Graph, Texture, Surface) ─────────────────────
        // Graph API stubs
        "cuGraphCreate" => Some(stubs::cuGraphCreate as *mut c_void),
        "cuGraphDestroy" => Some(stubs::cuGraphDestroy as *mut c_void),
        "cuGraphLaunch" => Some(stubs::cuGraphLaunch as *mut c_void),
        "cuGraphInstantiate" | "cuGraphInstantiate_v2" => {
            Some(stubs::cuGraphInstantiate as *mut c_void)
        }
        "cuGraphInstantiateWithFlags" => Some(stubs::cuGraphInstantiateWithFlags as *mut c_void),
        "cuGraphExecDestroy" => Some(stubs::cuGraphExecDestroy as *mut c_void),
        "cuGraphExecUpdate" | "cuGraphExecUpdate_v2" => {
            Some(stubs::cuGraphExecUpdate as *mut c_void)
        }
        "cuGraphAddKernelNode" | "cuGraphAddKernelNode_v2" => {
            Some(stubs::cuGraphAddKernelNode as *mut c_void)
        }
        "cuGraphAddMemcpyNode" => Some(stubs::cuGraphAddMemcpyNode as *mut c_void),
        "cuGraphAddMemsetNode" => Some(stubs::cuGraphAddMemsetNode as *mut c_void),
        "cuGraphAddHostNode" => Some(stubs::cuGraphAddHostNode as *mut c_void),
        "cuGraphAddChildGraphNode" => Some(stubs::cuGraphAddChildGraphNode as *mut c_void),
        "cuGraphAddEmptyNode" => Some(stubs::cuGraphAddEmptyNode as *mut c_void),
        "cuGraphAddEventRecordNode" => Some(stubs::cuGraphAddEventRecordNode as *mut c_void),
        "cuGraphAddEventWaitNode" => Some(stubs::cuGraphAddEventWaitNode as *mut c_void),
        "cuGraphUpload" => Some(stubs::cuGraphUpload as *mut c_void),
        "cuGraphNodeGetType" => Some(stubs::cuGraphNodeGetType as *mut c_void),
        "cuGraphGetRootNodes" => Some(stubs::cuGraphGetRootNodes as *mut c_void),
        "cuGraphGetNodes" => Some(stubs::cuGraphGetNodes as *mut c_void),
        "cuGraphGetEdges" | "cuGraphGetEdges_v2" => Some(stubs::cuGraphGetEdges as *mut c_void),
        "cuGraphAddDependencies" => Some(stubs::cuGraphAddDependencies as *mut c_void),
        "cuGraphRemoveDependencies" => Some(stubs::cuGraphRemoveDependencies as *mut c_void),
        "cuGraphClone" => Some(stubs::cuGraphClone as *mut c_void),
        "cuGraphNodeFindInClone" => Some(stubs::cuGraphNodeFindInClone as *mut c_void),
        "cuGraphKernelNodeGetParams" | "cuGraphKernelNodeGetParams_v2" => {
            Some(stubs::cuGraphKernelNodeGetParams as *mut c_void)
        }
        "cuGraphKernelNodeSetParams" | "cuGraphKernelNodeSetParams_v2" => {
            Some(stubs::cuGraphKernelNodeSetParams as *mut c_void)
        }
        "cuGraphExecKernelNodeSetParams" | "cuGraphExecKernelNodeSetParams_v2" => {
            Some(stubs::cuGraphExecKernelNodeSetParams as *mut c_void)
        }
        "cuStreamBeginCapture" | "cuStreamBeginCapture_v2" => {
            Some(stubs::cuStreamBeginCapture as *mut c_void)
        }
        "cuStreamEndCapture" => Some(stubs::cuStreamEndCapture as *mut c_void),
        "cuStreamIsCapturing" => Some(stubs::cuStreamIsCapturing as *mut c_void),
        "cuStreamGetCaptureInfo" | "cuStreamGetCaptureInfo_v2" | "cuStreamGetCaptureInfo_v3" => {
            Some(stubs::cuStreamGetCaptureInfo as *mut c_void)
        }
        "cuGraphInstantiateWithParams" => {
            Some(stubs::cuGraphInstantiateWithParams as *mut c_void)
        }
        "cuGraphAddNode" | "cuGraphAddNode_v2" => Some(stubs::cuGraphAddNode as *mut c_void),

        // Texture reference stubs
        "cuTexRefSetAddress" | "cuTexRefSetAddress_v2" => {
            Some(stubs::cuTexRefSetAddress as *mut c_void)
        }
        "cuTexRefSetAddress2D" | "cuTexRefSetAddress2D_v3" => {
            Some(stubs::cuTexRefSetAddress2D as *mut c_void)
        }
        "cuTexRefSetFormat" => Some(stubs::cuTexRefSetFormat as *mut c_void),
        "cuTexRefSetFlags" => Some(stubs::cuTexRefSetFlags as *mut c_void),
        "cuTexRefGetAddress" | "cuTexRefGetAddress_v2" => {
            Some(stubs::cuTexRefGetAddress as *mut c_void)
        }
        "cuTexRefGetFormat" => Some(stubs::cuTexRefGetFormat as *mut c_void),
        "cuTexRefSetFilterMode" => Some(stubs::cuTexRefSetFilterMode as *mut c_void),
        "cuTexRefSetAddressMode" => Some(stubs::cuTexRefSetAddressMode as *mut c_void),
        "cuTexRefGetFilterMode" => Some(stubs::cuTexRefGetFilterMode as *mut c_void),
        "cuTexRefGetAddressMode" => Some(stubs::cuTexRefGetAddressMode as *mut c_void),
        "cuTexRefSetArray" => Some(stubs::cuTexRefSetArray as *mut c_void),
        "cuTexRefGetArray" => Some(stubs::cuTexRefGetArray as *mut c_void),
        "cuTexRefSetMipmappedArray" => Some(stubs::cuTexRefSetMipmappedArray as *mut c_void),
        "cuTexRefGetMipmappedArray" => Some(stubs::cuTexRefGetMipmappedArray as *mut c_void),
        "cuTexRefSetMaxAnisotropy" => Some(stubs::cuTexRefSetMaxAnisotropy as *mut c_void),
        "cuTexRefGetMaxAnisotropy" => Some(stubs::cuTexRefGetMaxAnisotropy as *mut c_void),

        // Surface reference stubs
        "cuSurfRefSetArray" => Some(stubs::cuSurfRefSetArray as *mut c_void),
        "cuSurfRefGetArray" => Some(stubs::cuSurfRefGetArray as *mut c_void),

        // Texture/Surface object stubs
        "cuTexObjectCreate" => Some(stubs::cuTexObjectCreate as *mut c_void),
        "cuTexObjectDestroy" => Some(stubs::cuTexObjectDestroy as *mut c_void),
        "cuTexObjectGetResourceDesc" => Some(stubs::cuTexObjectGetResourceDesc as *mut c_void),
        "cuTexObjectGetTextureDesc" => Some(stubs::cuTexObjectGetTextureDesc as *mut c_void),
        "cuTexObjectGetResourceViewDesc" => {
            Some(stubs::cuTexObjectGetResourceViewDesc as *mut c_void)
        }
        "cuSurfObjectCreate" => Some(stubs::cuSurfObjectCreate as *mut c_void),
        "cuSurfObjectDestroy" => Some(stubs::cuSurfObjectDestroy as *mut c_void),
        "cuSurfObjectGetResourceDesc" => Some(stubs::cuSurfObjectGetResourceDesc as *mut c_void),

        // External memory/semaphore stubs
        "cuImportExternalMemory" => Some(stubs::cuImportExternalMemory as *mut c_void),
        "cuExternalMemoryGetMappedBuffer" => {
            Some(stubs::cuExternalMemoryGetMappedBuffer as *mut c_void)
        }
        "cuDestroyExternalMemory" => Some(stubs::cuDestroyExternalMemory as *mut c_void),
        "cuImportExternalSemaphore" => Some(stubs::cuImportExternalSemaphore as *mut c_void),
        "cuSignalExternalSemaphoresAsync" => {
            Some(stubs::cuSignalExternalSemaphoresAsync as *mut c_void)
        }
        "cuWaitExternalSemaphoresAsync" => {
            Some(stubs::cuWaitExternalSemaphoresAsync as *mut c_void)
        }
        "cuDestroyExternalSemaphore" => Some(stubs::cuDestroyExternalSemaphore as *mut c_void),

        // CUDA Array stubs
        "cuArrayCreate" | "cuArrayCreate_v2" => Some(stubs::cuArrayCreate as *mut c_void),
        "cuArrayDestroy" => Some(stubs::cuArrayDestroy as *mut c_void),
        "cuArray3DCreate" | "cuArray3DCreate_v2" => Some(stubs::cuArray3DCreate as *mut c_void),
        "cuArrayGetDescriptor" | "cuArrayGetDescriptor_v2" => {
            Some(stubs::cuArrayGetDescriptor as *mut c_void)
        }
        "cuArray3DGetDescriptor" | "cuArray3DGetDescriptor_v2" => {
            Some(stubs::cuArray3DGetDescriptor as *mut c_void)
        }
        "cuArrayGetSparseProperties" => Some(stubs::cuArrayGetSparseProperties as *mut c_void),
        "cuArrayGetMemoryRequirements" => Some(stubs::cuArrayGetMemoryRequirements as *mut c_void),
        "cuArrayGetPlane" => Some(stubs::cuArrayGetPlane as *mut c_void),
        "cuMipmappedArrayCreate" => Some(stubs::cuMipmappedArrayCreate as *mut c_void),
        "cuMipmappedArrayDestroy" => Some(stubs::cuMipmappedArrayDestroy as *mut c_void),
        "cuMipmappedArrayGetLevel" => Some(stubs::cuMipmappedArrayGetLevel as *mut c_void),
        "cuMipmappedArrayGetSparseProperties" => {
            Some(stubs::cuMipmappedArrayGetSparseProperties as *mut c_void)
        }
        "cuMipmappedArrayGetMemoryRequirements" => {
            Some(stubs::cuMipmappedArrayGetMemoryRequirements as *mut c_void)
        }

        // Deprecated module stubs
        "cuModuleGetTexRef" => Some(stubs::cuModuleGetTexRef as *mut c_void),
        "cuModuleGetSurfRef" => Some(stubs::cuModuleGetSurfRef as *mut c_void),

        // Misc stubs
        "cuGetExportTable" => Some(stubs::cuGetExportTable as *mut c_void),
        "cuFlushGPUDirectRDMAWrites" => Some(stubs::cuFlushGPUDirectRDMAWrites as *mut c_void),
        "cuStreamAddCallback" => Some(stubs::cuStreamAddCallback as *mut c_void),
        "cuLaunchHostFunc" => Some(stubs::cuLaunchHostFunc as *mut c_void),
        "cuOccupancyMaxPotentialBlockSize" => {
            Some(stubs::cuOccupancyMaxPotentialBlockSize as *mut c_void)
        }
        "cuOccupancyMaxPotentialBlockSizeWithFlags" => {
            Some(stubs::cuOccupancyMaxPotentialBlockSizeWithFlags as *mut c_void)
        }
        "cuMemHostRegister" | "cuMemHostRegister_v2" => {
            Some(stubs::cuMemHostRegister as *mut c_void)
        }
        "cuMemHostUnregister" => Some(stubs::cuMemHostUnregister as *mut c_void),

        // ── Not found ───────────────────────────────────────────
        _ => None,
    };

    match func_ptr {
        Some(ptr) => {
            *pfn = ptr;
            if !symbol_status.is_null() {
                *symbol_status = 1; // found
            }
            CUDA_SUCCESS
        }
        None => {
            *pfn = std::ptr::null_mut();
            if !symbol_status.is_null() {
                *symbol_status = 2; // not found
            }
            tracing::debug!("cuGetProcAddress: '{}' not found", name);
            CUDA_ERROR_NOT_FOUND
        }
    }
}

/// Simplified cuGetProcAddress (without v2 flags).
#[no_mangle]
pub unsafe extern "C" fn cuGetProcAddress(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
) -> CUresult {
    cuGetProcAddress_v2(symbol, pfn, cuda_version, flags, std::ptr::null_mut())
}
