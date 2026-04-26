<div align="center">
  <p align="center">

  <picture>
    <img alt="Flash Float JIT KERNEL" src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels/main/assets/img/FlashFloat.png" width="50%">

  </p>
  <h3>Flash Float Ultra Low Latency Hardware Aware Decoding JIT Kernel Library</h3>
  <a href="#cite-us">📝 Papers</a> | <a href="#QuickStart">🚀 Quick Start</a> | <a href="#support-dits">🎯 Supported Flash-Float-JIT-Kernel</a> | <a href="#dev-guide">📚 Dev Guide </a> | <a href="https://github.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels/discussions">📈  Discussion </a> | <a href="#Highlight">📝 Highlight </a></strong>
  <p></p>


</div>

This repository is complemntary of "FlashFloat" with JIT Kernels

<h2 id="highlight"> Highlight </h2>

- [🔥 ultra low latency topk , reduced maximum 50% 🚀 batch=1 latency for 1-M context 🎯!](#Ultra-Low-Latency-TopK-Indexer)


<h2 id="Ultra-Low-Latency-TopK-Indexer">🔥 Ultra Low Latency TopK Indexer</h2>

The introduction of the NSA (Neural Sparse Attention) mechanism in DeepSeek V3.2 has become pivotal for mitigating inference latency in long-context language modeling. While the NSA top-k indexer playing a critical role in reducing computational overhead for extended sequences[1], DeepSeek V4 [2] [3], recently, further pushes the context window limit to 1 million tokens where selecting top-2048 dimensions in sparse attention from upto 1-M context is prohibitive bottlenect upto 0.1 ms per layer per token in a sequence modeling task for agentic workflow.

Leveraging the latest SGLang (2026.3) as the benchmark, we investigate the root cause of this latency bottleneck: conventional throughput-optimized GPU designs suffer from low device utilization in low-batch scenarios due to insufficient inter-block coordination.[4] . For example, on-chip network have been maturely adopted for many years in processor hareware such as IPU, Cerebras and Groq, but only introduced into Hopper since 2022. By exploiting the limited on-chip communication capabilities of the Hopper architecture  (upto 8 blocks per cluster) through algorithm-hardware co-design, we achieve a 50% reduction in low-batch latency, demonstrating the effectiveness of synergistic optimization for long-context inference:

<picture>
  <img alt="dist-radix-radix-topk-indexer" src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels/main/assets/img/distributed-radix-topk-indexer.png">
</picture>

We hence propose Distributed Radix Sort via NoC to extremely reduce decoding latency in workload of low batch size, while with upto 1-M context length. First we compute historgram in parallel to reduce collision rates per block and then accumulate via NoC network before N-ways prefix sum and prove this is effectively method to reduce latency in a throughput oriented hardware. Second, we enhance the linear mapping properties for radix sort in **NSA** problem for reduction iteration. Instead of traditional top **8/11/13** bits of IEEE FP32, FP16 format, we redesign a linear mapping such that $bin(x) \large bin(y)$ , naturally deducing $x >= y$. With this linear mapping design, we greatly reduced per block elements dropped in the bin to determine the residule numbers. This further facilitate cache friendly visiting over 1-M context length : we hence enable less **SMEM** revisiting more elements.
<br/>

When remainder elements reduced to **8**/**16**, we can simply use **CAS** operations to performa a parallel sorting in few cycles. This further reduce the latency overhead in the last round.
<br/>

Previously, L2 cache was commonly used in NVGPU/AMD GPU to trackle the problem, for example in MoE Align Block Multi Block Execution Algorithm published 2025 [5], we tackle this problem by introducing mathematically equivalent **unaligned parallel prefix sum**. With distributed radix sort, we further prove that on-chip network can further reduce latency of our kernel, facilitating new design of algorithm and software for 1-M context.

## Reference

[1] DeepSeek-AI V3.2 (2024). DeepSeek-V32 Technical Report; Accessed on April 26, 2026

[2] DeepSeek-AI V4 (2026). DeepSeek-V4 Technical Report;  Accessed on April 26, 2026

[3] Dissecting DeepSeek V4 ：https://www.zhihu.com/question/2030963929510310856/answer/2031157557008541232?share_code=1nP5rOshEmo63&utm_psn=2031815957111419327; Accessed on April 26, 2026

[4] SGLang 2026.4 (0.5.10.post2.dev419+g635e922eb), classical throughput optimized design of TopK : https://github.com/sgl-project/sglang/blob/c7878dbb6ddfc9c6721b9db20a876f2718b0e955/sgl-kernel/csrc/elementwise/topk.cu#L448; Accessed on April 26 2026

[5] MoE Align and Sort : https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang