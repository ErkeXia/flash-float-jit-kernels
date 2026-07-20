import torch, triton
import jit_kernel.triton3_4.symm_gemm as s


SEED = 42


def test(m=2048):

    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    xq = torch.arange(m, dtype=torch.float16, device="cuda").view(1, -1) / (
        m - 1
    ) + torch.arange(m, dtype=torch.float16, device="cuda").view(-1, 1) / (m - 1)

    x_fp8 = xq.to(torch.float8_e4m3fn)

    xs_0 = torch.ones((m, triton.cdiv(m, 128)), dtype=torch.float32, device="cuda")
    xs_1 = torch.ones(
        (triton.cdiv(m, 128), triton.cdiv(m, 128)), dtype=torch.float32, device="cuda"
    )

    out = torch.empty((m, m), dtype=torch.float16, device="cuda")

    sentinel = -1234.0

    # first call: populate TVM-FFI cache
    print("first call : ...")
    out.fill_(sentinel)
    s.thunder_moun_gemm(x_fp8, x_fp8, xs_0, xs_1, out=out)
    torch.cuda.synchronize()
    print("first sentinel remaining:", (out == sentinel).sum().item())

    # second call: cached TVM-FFI path only
    print("second call : ...")
    out.fill_(sentinel)
    s.thunder_moun_gemm(x_fp8, x_fp8, xs_0, xs_1, out=out)
    torch.cuda.synchronize()
    remaining = (out == sentinel).sum().item()

    print("cached sentinel remaining:", remaining)
    print("total elements:", out.numel())


if __name__ == "__main__":
    test()