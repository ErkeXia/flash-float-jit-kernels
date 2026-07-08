import triton

import tvm_ffi

from packaging import version

import inspect
import triton.language as tl

def is_constexpr_param(param):
    ann = param.annotation
    if ann is inspect._empty:
        return False
    return ann is tl.constexpr or getattr(ann, "__name__", "") == "constexpr" or "constexpr" in str(ann)

def cpp_host_type(name: str) -> str:
    if "ptr" in name:
        return "TensorView"
    if "use_" in name or "is_" in name:
        return "bool"
    return "int32_t"


def kernel_arg_type(name: str) -> str:
    if "ptr" in name:
        return "void*"
    if "use_" in name or "is_" in name:
        return "bool"
    return "int32_t"

def kernel_arg_assignment(name: str) -> str:
    if "ptr" in name:
        return f"kargs.{name} = {name}.data_ptr();\n"
    return f"kargs.{name} = {name};\n"

def runtime_arg_decl(name: str) -> str:
    var = f"{name}_arg"
    if "ptr" in name:
        return f"void* {var} = {name}.data_ptr();\n"
    if "use_" in name or "is_" in name:
        return f"bool {var} = {name};\n"
    return f"int32_t {var} = {name};\n"

# Triton 3.4
def generate_tvm_ffi_source(compiled_kernel, kernel_name, debug=False):
    # Triton 3.5+
    if version.parse(triton.__version__) >= version.parse("3.5"):
        raise Exception("TVM FFI Not Implemented for Triton 3.5+ due to the change of metadata structure, need to update the parsing logic accordingly.")
    
    if debug:
        print(compiled_kernel.metadata)
        print(compiled_kernel.asm["ptx"].split(".entry", 1)[1].split(")", 1)[0])

    arg_names = compiled_kernel.src.fn.arg_names
    signature = compiled_kernel.src.fn.signature 

    # cpp_params = ["int64_t cubin_ptr_addr"]   # FFI args list
    cpp_params = []
    cpp_arg_names = []

    launch_args_def = []  # cuLaunchKernel pointer args list
    launch_args = []  # cuLaunchKernel args list
    launch_arg_names = []

    constants = {}
    constants_set = set()
    for key, val in compiled_kernel.src.constants.items():
        name = arg_names[key[0]]

        # if "stride_" in name:
        #     if debug:
        #         print(f"{name} is not regarded as constant")
        #     continue

        constants[name] = (key[0], val)
        constants_set.add(name)

    if debug:
        print("arg_names : ", arg_names)
        print("constants : ", constants)

    for name, param in signature.parameters.items():
        if is_constexpr_param(param):
            continue

        cpp_arg_names.append(name)

        if name not in constants_set:
            launch_arg_names.append(name)
            
    cpp_params = [f"{cpp_host_type(name)} {name}" for name in cpp_arg_names]
    cpp_params_str = ", ".join(cpp_params + ["int32_t grid_x", "int32_t grid_y", "int32_t grid_z"])
    
    launch_args_def = [
        f"{kernel_arg_type(name)} {name};\n"
        for name in launch_arg_names
    ]

    launch_args = [
        kernel_arg_assignment(name)
        for name in launch_arg_names
    ]
    
    launch_args_def.append("CUdeviceptr global_scratch;\n")
    launch_args.append("kargs.global_scratch = 0;\n")

    launch_args_def_str = "".join(launch_args_def)
    launch_args_str = "".join(launch_args)
    
    runtime_arg_decls = [
        runtime_arg_decl(name)
        for name in launch_arg_names
    ]

    runtime_arg_ptrs = [
        f"&{name}_arg"
        for name in launch_arg_names
    ]

    runtime_arg_decls.append("CUdeviceptr global_scratch_arg = 0;\n")
    runtime_arg_ptrs.append("&global_scratch_arg")

    runtime_arg_decls_str = "".join(runtime_arg_decls)
    runtime_arg_ptrs_str = ",\n        ".join(runtime_arg_ptrs)

    if debug:
        print(f"cpp_params_str : {cpp_params_str}")
        print(f"launch_args_def_str : {launch_args_def_str}")
        print(f"launch_args_str : {launch_args_str}")

    META = compiled_kernel.metadata
    
    global_scratch_size = getattr(META, "global_scratch_size", 0)
    global_scratch_align = getattr(META, "global_scratch_align", 1)

    num_warps = META.num_warps # compiled_kernel.num_warps
    shared_mem_size = META.shared

    arch = META.target.arch
    WARP_SIZE = META.target.warp_size

    source = f"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define USE_TVM_FFI_LAUNCH_CONVENTION 0

TVM_FFI_EMBED_CUBIN(triton_cubin);

namespace triton_loader {{
using namespace tvm::ffi;

// NOTE (yiakwy) : TVM's official method does not handle alignment issue, hence I use this method to escape alignment problem. 
// We can also consider to modify TVM's official method to support alignment in the future if necessary.
struct KernelArgs {{
    {launch_args_def_str};
}};

void {kernel_name}_launcher({cpp_params_str}) {{
    static auto launcher = TVM_FFI_EMBED_CUBIN_GET_KERNEL(triton_cubin, "{kernel_name}");

    // construct Triton arguments list
    KernelArgs kargs;

    {launch_args_str};
    
    //runtime-api
    {runtime_arg_decls_str}

    void* args[] = {{
            {runtime_arg_ptrs_str}
    }};
        
    cudaGetLastError();

    int shared_mem_bytes = {shared_mem_size};
    // cuDeviceGetAttribute(&shared_mem_bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 0);
    
    auto kernel = launcher.GetHandle();
    // CUfunction kernel = *(reinterpret_cast<CUfunction*>(&launcher));
    // CUresult attr_result = cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (unsigned int)shared_mem_bytes);
    
    /* 
    if (attr_result != CUDA_SUCCESS) {{
        TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
            static_cast<::tvm::ffi::cuda_api::ResultType>(attr_result)
        );
    }}
    */
    
    DLDevice device = {arg_names[0]}.device();
    
    ::tvm::ffi::cuda_api::StreamHandle stream =
        static_cast<::tvm::ffi::cuda_api::StreamHandle>(
            TVMFFIEnvGetStream(device.device_type, device.device_id));

    auto device_handle =
        ::tvm::ffi::cuda_api::GetDeviceHandle(device.device_id);

    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
        ::tvm::ffi::cuda_api::SetKernelMaxDynamicSharedMem(
            kernel,
            shared_mem_bytes,
            device_handle)
    );
    
    // cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

    tvm::ffi::dim3 grid((unsigned int)grid_x, (unsigned int)grid_y, (unsigned int)grid_z);
    tvm::ffi::dim3 block({num_warps * WARP_SIZE}, 1, 1);

#if USE_TVM_FFI_LAUNCH_CONVENTION
    // Launch Kernel : old version does not support to pass shared_mem_bytes, so here is my workaround
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(launcher.Launch(args, grid, block, stream, shared_mem_bytes));


#else
    // see https://github.com/apache/tvm-ffi/blob/d73a26783488430986fd855ae67ff7a9fb016413/include/tvm/ffi/extra/cuda/internal/unified_api.h#L115
    /*  
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(static_cast<::tvm::ffi::cuda_api::ResultType>(
        cuLaunchKernel(kernel, 
          grid.x, grid.y, grid.z, 
          block.x, block.y, block.z, 
          (unsigned int)shared_mem_bytes, stream, nullptr, config);
    ));
    */


#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
    size_t kargs_size = sizeof(KernelArgs);
    void* config[] = {{
        CU_LAUNCH_PARAM_BUFFER_POINTER, &kargs,
        CU_LAUNCH_PARAM_BUFFER_SIZE,    &kargs_size,
        CU_LAUNCH_PARAM_END
    }};  

    CUresult result = cuLaunchKernel(
          reinterpret_cast<CUfunction>(kernel), 
          grid.x, grid.y, grid.z, 
          block.x, block.y, block.z, 
          (unsigned int)shared_mem_bytes, stream, nullptr, config);

    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(result);
#else
    cudaError_t result = cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel),
        ::dim3(grid.x, grid.y, grid.z),
        ::dim3(block.x, block.y, block.z),
        args,
        (size_t)shared_mem_bytes,
        stream);

    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(result);
#endif

    if (result != CUDA_SUCCESS) {{
        TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(static_cast<::tvm::ffi::cuda_api::ResultType>(result));
    }}

    // cudaDeviceSynchronize();
#endif
}}

}} // namespace triton_loader

TVM_FFI_DLL_EXPORT_TYPED_FUNC({kernel_name}, triton_loader::{kernel_name}_launcher);
"""
    return source, constants 