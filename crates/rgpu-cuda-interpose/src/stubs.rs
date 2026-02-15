//! Stub implementations for unsupported CUDA APIs.
//!
//! These functions return CUDA_ERROR_NOT_SUPPORTED (801) for:
//! - CUDA Graph APIs
//! - Legacy Texture/Surface reference APIs
//! - Texture/Surface object APIs
//! - External memory/semaphore APIs
//! - Callback-based functions (cannot work over network)
//! - Other miscellaneous unsupported functions

use std::ffi::c_int;

type CUresult = c_int;
const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
const CUDA_ERROR_NOT_FOUND: CUresult = 500;
const CUDA_SUCCESS: CUresult = 0;

// ── Graph API Stubs ──────────────────────────────────────────────

// We can't use variadic C functions in stable Rust easily, so define each stub explicitly.
// All take arbitrary arguments and return CUDA_ERROR_NOT_SUPPORTED.

#[no_mangle] pub unsafe extern "C" fn cuGraphCreate(_graph: *mut *mut std::ffi::c_void, _flags: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphDestroy(_graph: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphLaunch(_exec: *mut std::ffi::c_void, _stream: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphInstantiate(_exec: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _nodes: *mut *mut std::ffi::c_void, _log: *mut i8, _buf_size: usize) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphInstantiateWithFlags(_exec: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _flags: u64) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphInstantiateWithParams(_exec: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _params: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphExecDestroy(_exec: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphExecUpdate(_exec: *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _result: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddKernelNode(_node: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _deps: *const *mut std::ffi::c_void, _num_deps: usize, _params: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddMemcpyNode(_node: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _deps: *const *mut std::ffi::c_void, _num_deps: usize, _params: *const std::ffi::c_void, _ctx: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddMemsetNode(_node: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _deps: *const *mut std::ffi::c_void, _num_deps: usize, _params: *const std::ffi::c_void, _ctx: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddHostNode(_node: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _deps: *const *mut std::ffi::c_void, _num_deps: usize, _params: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddChildGraphNode(_node: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _deps: *const *mut std::ffi::c_void, _num_deps: usize, _child: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddEmptyNode(_node: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _deps: *const *mut std::ffi::c_void, _num_deps: usize) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddEventRecordNode(_node: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _deps: *const *mut std::ffi::c_void, _num_deps: usize, _event: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddEventWaitNode(_node: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _deps: *const *mut std::ffi::c_void, _num_deps: usize, _event: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphUpload(_exec: *mut std::ffi::c_void, _stream: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphNodeGetType(_node: *mut std::ffi::c_void, _type_out: *mut c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphGetRootNodes(_graph: *mut std::ffi::c_void, _nodes: *mut *mut std::ffi::c_void, _num: *mut usize) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphGetNodes(_graph: *mut std::ffi::c_void, _nodes: *mut *mut std::ffi::c_void, _num: *mut usize) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphGetEdges(_graph: *mut std::ffi::c_void, _from: *mut *mut std::ffi::c_void, _to: *mut *mut std::ffi::c_void, _num: *mut usize) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddDependencies(_graph: *mut std::ffi::c_void, _from: *const *mut std::ffi::c_void, _to: *const *mut std::ffi::c_void, _num: usize) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphRemoveDependencies(_graph: *mut std::ffi::c_void, _from: *const *mut std::ffi::c_void, _to: *const *mut std::ffi::c_void, _num: usize) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphClone(_clone: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphNodeFindInClone(_clone_node: *mut *mut std::ffi::c_void, _node: *mut std::ffi::c_void, _clone_graph: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphKernelNodeGetParams(_node: *mut std::ffi::c_void, _params: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphKernelNodeSetParams(_node: *mut std::ffi::c_void, _params: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphExecKernelNodeSetParams(_exec: *mut std::ffi::c_void, _node: *mut std::ffi::c_void, _params: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuGraphAddNode(_node: *mut *mut std::ffi::c_void, _graph: *mut std::ffi::c_void, _deps: *const *mut std::ffi::c_void, _num_deps: usize, _params: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }

// Stream capture stubs
#[no_mangle] pub unsafe extern "C" fn cuStreamBeginCapture(_stream: *mut std::ffi::c_void, _mode: c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuStreamEndCapture(_stream: *mut std::ffi::c_void, _graph: *mut *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuStreamIsCapturing(_stream: *mut std::ffi::c_void, _status: *mut c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuStreamGetCaptureInfo(_stream: *mut std::ffi::c_void, _status: *mut c_int, _id: *mut u64) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }

// ── Texture Reference Stubs ─────────────────────────────────────

#[no_mangle] pub unsafe extern "C" fn cuTexRefSetAddress(_offset: *mut usize, _tex: *mut std::ffi::c_void, _dptr: u64, _bytes: usize) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefSetAddress2D(_tex: *mut std::ffi::c_void, _desc: *const std::ffi::c_void, _dptr: u64, _pitch: usize) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefSetFormat(_tex: *mut std::ffi::c_void, _fmt: c_int, _num_channels: c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefSetFlags(_tex: *mut std::ffi::c_void, _flags: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefGetAddress(_dptr: *mut u64, _tex: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefGetFormat(_fmt: *mut c_int, _num_channels: *mut c_int, _tex: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefSetFilterMode(_tex: *mut std::ffi::c_void, _mode: c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefSetAddressMode(_tex: *mut std::ffi::c_void, _dim: c_int, _mode: c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefGetFilterMode(_mode: *mut c_int, _tex: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefGetAddressMode(_mode: *mut c_int, _tex: *mut std::ffi::c_void, _dim: c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefSetArray(_tex: *mut std::ffi::c_void, _array: *mut std::ffi::c_void, _flags: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefGetArray(_array: *mut *mut std::ffi::c_void, _tex: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefSetMipmappedArray(_tex: *mut std::ffi::c_void, _array: *mut std::ffi::c_void, _flags: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefGetMipmappedArray(_array: *mut *mut std::ffi::c_void, _tex: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefSetMaxAnisotropy(_tex: *mut std::ffi::c_void, _max: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexRefGetMaxAnisotropy(_max: *mut u32, _tex: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }

// ── Surface Reference Stubs ────────────────────────────────────

#[no_mangle] pub unsafe extern "C" fn cuSurfRefSetArray(_surf: *mut std::ffi::c_void, _array: *mut std::ffi::c_void, _flags: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuSurfRefGetArray(_array: *mut *mut std::ffi::c_void, _surf: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }

// ── Texture/Surface Object Stubs ────────────────────────────────

#[no_mangle] pub unsafe extern "C" fn cuTexObjectCreate(_obj: *mut u64, _res_desc: *const std::ffi::c_void, _tex_desc: *const std::ffi::c_void, _view_desc: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexObjectDestroy(_obj: u64) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexObjectGetResourceDesc(_desc: *mut std::ffi::c_void, _obj: u64) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexObjectGetTextureDesc(_desc: *mut std::ffi::c_void, _obj: u64) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuTexObjectGetResourceViewDesc(_desc: *mut std::ffi::c_void, _obj: u64) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuSurfObjectCreate(_obj: *mut u64, _res_desc: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuSurfObjectDestroy(_obj: u64) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuSurfObjectGetResourceDesc(_desc: *mut std::ffi::c_void, _obj: u64) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }

// ── External Memory/Semaphore Stubs ─────────────────────────────

#[no_mangle] pub unsafe extern "C" fn cuImportExternalMemory(_ext_mem: *mut *mut std::ffi::c_void, _desc: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuExternalMemoryGetMappedBuffer(_dptr: *mut u64, _ext_mem: *mut std::ffi::c_void, _desc: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuDestroyExternalMemory(_ext_mem: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuImportExternalSemaphore(_ext_sem: *mut *mut std::ffi::c_void, _desc: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuSignalExternalSemaphoresAsync(_sems: *const *mut std::ffi::c_void, _params: *const std::ffi::c_void, _num: u32, _stream: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuWaitExternalSemaphoresAsync(_sems: *const *mut std::ffi::c_void, _params: *const std::ffi::c_void, _num: u32, _stream: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuDestroyExternalSemaphore(_ext_sem: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }

// ── Callback-based Function Stubs ───────────────────────────────

#[no_mangle] pub unsafe extern "C" fn cuStreamAddCallback(_stream: *mut std::ffi::c_void, _callback: *mut std::ffi::c_void, _user_data: *mut std::ffi::c_void, _flags: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuLaunchHostFunc(_stream: *mut std::ffi::c_void, _fn_ptr: *mut std::ffi::c_void, _user_data: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuOccupancyMaxPotentialBlockSize(_min_grid: *mut c_int, _block_size: *mut c_int, _func: *mut std::ffi::c_void, _callback: *mut std::ffi::c_void, _dyn_smem: usize, _block_limit: c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuOccupancyMaxPotentialBlockSizeWithFlags(_min_grid: *mut c_int, _block_size: *mut c_int, _func: *mut std::ffi::c_void, _callback: *mut std::ffi::c_void, _dyn_smem: usize, _block_limit: c_int, _flags: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }

// ── CUDA Array Stubs ─────────────────────────────────────────────

#[no_mangle] pub unsafe extern "C" fn cuArrayCreate(_array: *mut *mut std::ffi::c_void, _desc: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuArrayDestroy(_array: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuArray3DCreate(_array: *mut *mut std::ffi::c_void, _desc: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuArrayGetDescriptor(_desc: *mut std::ffi::c_void, _array: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuArray3DGetDescriptor(_desc: *mut std::ffi::c_void, _array: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuArrayGetSparseProperties(_props: *mut std::ffi::c_void, _array: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuArrayGetMemoryRequirements(_reqs: *mut std::ffi::c_void, _array: *mut std::ffi::c_void, _device: c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuArrayGetPlane(_plane_array: *mut *mut std::ffi::c_void, _array: *mut std::ffi::c_void, _plane_idx: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuMipmappedArrayCreate(_array: *mut *mut std::ffi::c_void, _desc: *const std::ffi::c_void, _num_levels: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuMipmappedArrayDestroy(_array: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuMipmappedArrayGetLevel(_level: *mut *mut std::ffi::c_void, _array: *mut std::ffi::c_void, _level_idx: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuMipmappedArrayGetSparseProperties(_props: *mut std::ffi::c_void, _array: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuMipmappedArrayGetMemoryRequirements(_reqs: *mut std::ffi::c_void, _array: *mut std::ffi::c_void, _device: c_int) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }

// ── Deprecated Module Stubs ─────────────────────────────────────

#[no_mangle] pub unsafe extern "C" fn cuModuleGetTexRef(_tex: *mut *mut std::ffi::c_void, _module: *mut std::ffi::c_void, _name: *const i8) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuModuleGetSurfRef(_surf: *mut *mut std::ffi::c_void, _module: *mut std::ffi::c_void, _name: *const i8) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }

// ── Miscellaneous Stubs ─────────────────────────────────────────

#[no_mangle] pub unsafe extern "C" fn cuGetExportTable(_table: *mut *const std::ffi::c_void, _id: *const std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_FOUND }
#[no_mangle] pub unsafe extern "C" fn cuFlushGPUDirectRDMAWrites(_target: c_int, _scope: c_int) -> CUresult { CUDA_SUCCESS }
#[no_mangle] pub unsafe extern "C" fn cuMemHostRegister(_p: *mut std::ffi::c_void, _byte_size: usize, _flags: u32) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
#[no_mangle] pub unsafe extern "C" fn cuMemHostUnregister(_p: *mut std::ffi::c_void) -> CUresult { CUDA_ERROR_NOT_SUPPORTED }
