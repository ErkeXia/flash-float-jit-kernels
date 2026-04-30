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

- April 27 2026, [🔥 ultra low latency topk , reduced maximum 50% 🚀 batch=1 latency for 1-M context 🎯!](#Ultra-Low-Latency-TopK-Indexer)


<h2 id="Ultra-Low-Latency-TopK-Indexer">🔥 Ultra Low Latency TopK Indexer</h2>

The introduction of the NSA (Native Sparse Attention) mechanism in DeepSeek V3.2 has become pivotal for mitigating inference latency in long-context language modeling. While the NSA top-k indexer playing a critical role in reducing computational overhead for DeepSeek V32 long context sequence modeling task[1], DeepSeek V4 [2] [3], recently, further pushes the context window limit to 1 million tokens where selecting top-2048 dimensions in hybrid sparse attention along from upto 1-M context is a prohibitive bottlenect upto **0.1** ms per layer per query token for agentic workflow.

Leveraging the latest SGLang (2026.3) as the benchmark, we investigated the root causes of this latency bottleneck: kernels for conventional throughput-optimized GPU designs suffer from low device utilization in low-batch, yet long context decoding scenarios due to insufficient inter-block coordination [4]. For example, on-chip network have been maturely adopted for many years in processor hareware such as IPU, Cerebras and Groq, but only being introduced into Hopper since 2022. By exploiting the limited on-chip communication capabilities of the Hopper architecture  (upto 8 blocks per cluster) through hardware-aware alorithm-hardware co-design, we achieve a **50%** latency reduction in low-batch yet long context decoding scenarios, demonstrating the effectiveness of synergistic optimization and **NoC** for long-context inference:

<picture>
  <img alt="dist-radix-radix-topk-indexer" src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels/main/assets/img/distributed-radix-topk-indexer.png">
</picture>

We hence propose **Distributed Radix Sort via NoC** to extremely reduce decoding latency in workload of low batch size, yet with upto 1-M context length. 

- First we compute historgram in parallel to reduce collision rates per block and then accumulate the histogram via NoC network before N-ways prefix sum and prove this is an effective method to reduce latency for a throughput oriented hardware design. 

- Second, we enhance the linear mapping properties for radix sort in **NSA** problem for reduction of radix sorting iterations; insteadd of traditional top **8/11/13** bits of IEEE FP32, FP16 format, we redesign the linear mapping such that $bin(x) >= bin(y)$, naturally deducing $x >= y$. 

  With this linear mapping design, we greatly reduced per block elements dropped in the threshold bin in redix sorting scheme and greatly reduce the residual numbers in later rounds.
  
  This further facilitate cache friendly re-visiting over 1-M context length : we hence enable less **SMEM** in revisiting more elements.
<br/>

- Finally, when remainder elements reduced to **8**/**16**, we can simply use **CAS** operations to performa a **neat parallel sorting** in few cycles. This further reduce the latency overhead in the last round.
<br/>

Previously, L2 cache was commonly used in NVGPU/AMD GPU to trackle the problem, for example in MoE Align Block Multi Block Execution Algorithm published 2025 [5], we tackle this problem by introducing mathematically equivalent **unaligned parallel prefix sum**. With distributed radix sort, we further prove that on-chip network can further reduce latency of our kernel, facilitating new design of algorithm and software for 1-M context.

#### Compared to other works at the the time article is composing

###### TLE DSL Topk (April 17, 2026)
Triton-TLE (Triton Language Extension) [TopK](https://github.com/flagos-ai/FlagTree/blob/f9a8d23602a65ec5c1af3b117e1faa46fe6f63b7/python/tutorials/tle/deepseek_v32/01-topk_selector.py#L3055) sits on the Pareto frontier of the prductivity and performance, filling the gap between our CUDA and other high-level DSL such as triton-Gluon and TileLang. It is highly efficient and elegant to utilize the DSHMEM to reduce the cross blocks communications lantency.

The triton extension introduce the semantics explicitly visiting remote (tle.remote) blocks within the clusters via block-level device mesh (**tle.device_mesh**), cluster barriers (**tle.distributed_barrier**) and close loop on chip processing. While our native CUDA implementation offers peak performance, this DSL drastically simplifies the implementation complexity compared to manual CUDA coding.

###### TileLang
Our CUDA implementation stems directly from the SGLang-optimized variant (fast_topk_v2) of the tileLang implementation. Built natively on the TVM FFI, tileLang significantly reduces CPU overhead and facilitates the rapid adoption of tiling-based programming advantages.

However, both the tileLang DSL and its generated CUDA kernels lack optimizations for cross-block communication via the Network-on-Chip (NoC). By relying on global memory for inter-block synchronization rather than hardware-accelerated features like DSMEM (Distributed Shared Memory), they exhibit significantly higher latency in our benchmarks compared to our CUDA implementation.

On the other hand, tileLang excels in its pipelining mechanism, which enables efficient I/O-compute overlap. This is particularly advantageous in scenarios where the computational load is heavy enough to hide the memory latency of TopK operations. We will further explore these trade-offs and our integration strategies in the discussion of our ThunderMuon optimizer.

###### Triton Gluon


###### Flashinfer TopK


###### TRT-LLM (April 27, 2026)


## Reference

[1] DeepSeek-AI V3.2 (2025.12). DeepSeek-V32 Technical Report: "Pushing the Frontier of Open Large Language Models", arXiv:2512.02556; Accessed on April 26, 2026

[2] DeepSeek-AI V4 (2026). DeepSeek-V4 Technical Report: "Towards Highly Efficient Millon Token Context Intelligence", https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf;  Accessed on April 26, 2026

[3] Dissecting DeepSeek V4 ：https://www.zhihu.com/question/2030963929510310856/answer/2031157557008541232?share_code=1nP5rOshEmo63&utm_psn=2031815957111419327; Accessed on April 26, 2026

[4] SGLang 2026.4 (0.5.10.post2.dev419+g635e922eb), classical throughput optimized design of TopK : https://github.com/sgl-project/sglang/blob/c7878dbb6ddfc9c6721b9db20a876f2718b0e955/sgl-kernel/csrc/elementwise/topk.cu#L448; Accessed on April 26 2026

[5] MoE Align and Sort (2025.3), https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang, LEI (yiak.wy@gmail.com); Accessed on April 26 2026


## Citation

If you use this codebase, or otherwise find our work valuable, please cite Gram Newton-Schulz:

```bibtex
@misc{DistRadixTopK2026,
  title   = {Ultra Low Latency Distributed Radix TopK Indexer Via NoC},
  author  = {LEI WANG, Hui Guo, Hao Gu, Bei Liu, Sirui Han, Wei Xue, Yike Guo},
  year    = {2026},
  url     = {https://github.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels}
}
```