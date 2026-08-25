import torch

import jit_kernel.triton3_5.gluon.symm_gemm as s


SEED = 42
SENTINEL = -1000.0


def _check_result(out, expected):
    assert not torch.any(out == SENTINEL), "Gluon XXT did not overwrite all outputs"
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


def test(m=2048, k=2048):
    torch.manual_seed(SEED)

    x = torch.randn((m, k), dtype=torch.float32, device="cuda")
    x = (x / (k**0.5)).to(torch.bfloat16)
    expected = torch.matmul(x.float(), x.float().T)

    symm_gemm_op = s.GluonXXT()
    s.tvm_ffi_modules["XXT"] = None

    # Cache miss: compile the Gluon kernel and build the TVM-FFI launcher.
    out = torch.full((m, m), SENTINEL, dtype=x.dtype, device=x.device)
    returned = symm_gemm_op(x, out=out, use_tvm_ffi=True)
    torch.cuda.synchronize()
    assert returned is out
    _check_result(out, expected)

    cache = s.tvm_ffi_modules["XXT"]
    assert cache is not None and len(cache) == 1
    cached_module = next(iter(cache.values()))[0]

    # Cache hit with the same output tensor.
    out.fill_(SENTINEL)
    returned = symm_gemm_op(x, out=out, use_tvm_ffi=True)
    torch.cuda.synchronize()
    assert returned is out
    _check_result(out, expected)
    assert len(s.tvm_ffi_modules["XXT"]) == 1
    assert next(iter(s.tvm_ffi_modules["XXT"].values()))[0] is cached_module

    # Cache hit with a different output tensor. The TensorMap must be rebuilt
    # with the new data pointer even though the compiled module is reused.
    out2 = torch.full_like(out, SENTINEL)
    returned = symm_gemm_op(x, out=out2, use_tvm_ffi=True)
    torch.cuda.synchronize()
    assert returned is out2
    _check_result(out2, expected)
    assert len(s.tvm_ffi_modules["XXT"]) == 1
    assert next(iter(s.tvm_ffi_modules["XXT"].values()))[0] is cached_module


if __name__ == "__main__":
    test()
