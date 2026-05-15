import triton

import tvm_ffi

from packaging import version

# Triton 3.4
def generate_tvm_ffi_source(compiled_kernel, kernel_name, debug=False):
    # Triton 3.5+
    if version.parse(triton.__version__) >= version.parse("3.5"):
        raise Exception("TVM FFI Not Implemented for Triton 3.5+ due to the change of metadata structure, need to update the parsing logic accordingly.")
    
    arg_names = compiled_kernel.src.fn.arg_names
    signature = compiled_kernel.src.fn.signature 

    # cpp_params = ["int64_t cubin_ptr_addr"]   # FFI args list
    cpp_params = []

    launch_args_def = []  # cuLaunchKernel pointer args list
    launch_args = []  # cuLaunchKernel args list

    constants = {}
    constants_set = set()
    for key, val in compiled_kernel.src.constants.items():
        name = arg_names[key[0]]

        if "stride_" in name:
            if debug:
                print(f"{name} is not regarded as constant")
            continue

        constants[name] = (key[0], val)
        constants_set.add(name)

    if debug:
        print("arg_names : ", arg_names)
        print("constants : ", constants)

    for name, param in signature.parameters.items():
        if name in constants_set:
            continue

        if "ptr" in name:
            cpp_params.append(f"TensorView {name}")
            launch_args_def.append(f"void* {name};\n")
            launch_args.append(f"kargs.{name} = {name}.data_ptr();\n")
        elif "use_" in name or "is_" in name:
            cpp_params.append(f"bool {name}")
            launch_args_def.append(f"bool {name};\n")
            launch_args.append(f"kargs.{name} = {name};\n")
        else:
            cpp_params.append(f"int32_t {name}")
            launch_args_def.append(f"int32_t {name};\n")
            launch_args.append(f"kargs.{name} = {name};\n")
            

    cpp_params_str = ", ".join(cpp_params + ["int32_t grid_x", "int32_t grid_y", "int32_t grid_z"])

    launch_args_def_str = "".join(launch_args_def)
    launch_args_str = "".join(launch_args)

    if debug:
        print(f"cpp_params_str : {cpp_params_str}")
        print(f"launch_args_def_str : {launch_args_def_str}")
        print(f"launch_args_str : {launch_args_str}")

    META = compiled_kernel.metadata

    num_warps = META.num_warps # compiled_kernel.num_warps
    shared_mem_size = 24 * 1024 # META.shared

    arch = META.target.arch
    WARP_SIZE = META.target.warp_size

    source = f"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cuda_runtime.h>

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
    
    cudaGetLastError();

    int shared_mem_bytes = {shared_mem_size};
    cuDeviceGetAttribute(&shared_mem_bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 0);
    
    CUfunction kernel = *(reinterpret_cast<CUfunction*>(&launcher));
    cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (unsigned int)shared_mem_bytes);
    
    DLDevice device = {arg_names[0]}.device();
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

    tvm::ffi::dim3 grid((unsigned int)grid_x, (unsigned int)grid_y, (unsigned int)grid_z);
    tvm::ffi::dim3 block({num_warps * WARP_SIZE}, 1, 1);

#if USE_TVM_FFI_LAUNCH_CONVENTION
    // Launch Kernel : old version does not support to pass shared_mem_bytes, so here is my workaround
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(launcher.Launch(args, grid, block, stream, shared_mem_bytes));


#else
    size_t kargs_size = sizeof(KernelArgs);
    void* config[] = {{
        CU_LAUNCH_PARAM_BUFFER_POINTER, &kargs,
        CU_LAUNCH_PARAM_BUFFER_SIZE,    &kargs_size,
        CU_LAUNCH_PARAM_END
    }};  

    // see https://github.com/apache/tvm-ffi/blob/d73a26783488430986fd855ae67ff7a9fb016413/include/tvm/ffi/extra/cuda/internal/unified_api.h#L115
    /*  
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(static_cast<::tvm::ffi::cuda_api::ResultType>(
        cuLaunchKernel(kernel, 
          grid.x, grid.y, grid.z, 
          block.x, block.y, block.z, 
          (unsigned int)shared_mem_bytes, stream, nullptr, config);
    ));
    */

    CUresult result = cuLaunchKernel(kernel, 
          grid.x, grid.y, grid.z, 
          block.x, block.y, block.z, 
          (unsigned int)shared_mem_bytes, stream, nullptr, config);

    if (result != CUDA_SUCCESS) {{
        // TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(static_cast<::tvm::ffi::cuda_api::ResultType>(result));
    }}

    // cudaDeviceSynchronize();
#endif
}}

}} // namespace triton_loader

TVM_FFI_DLL_EXPORT_TYPED_FUNC({kernel_name}, triton_loader::{kernel_name}_launcher);
"""
    return source, constants 