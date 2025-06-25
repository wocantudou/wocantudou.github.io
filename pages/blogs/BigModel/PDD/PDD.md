![PDD](BigModel/PDD/PDD.png)
# 大模型效率部署之Prefill-Decode分离

## Prefill 与 Decode 阶段定义与流程

LLM 推理分为两个阶段：**预填充（Prefill）**和**解码（Decode）**。在 Prefill 阶段，模型将完整地处理用户输入的所有提示词（prompt），一次性完成前向计算，生成隐藏层的键值对缓存（KV 缓存），并输出生成序列的第一个 Token。这一阶段相当于用矩阵乘法并行计算整个输入序列，高效地利用了 GPU 的算力（**算力密集型**）。进入 Decode 阶段后，模型开始自回归地逐步生成后续输出——每次仅处理一个新 Token，并利用前面累积的 KV 缓存计算注意力，产生下一个输出并更新 KV 缓存。解码阶段是一个迭代过程：每生成一个 Token，模型就将其附加到当前序列中，再做一次前向计算；直到生成结束标记或达到指定长度。与 Prefill 相比，Decode 多次执行小规模的前向运算，计算负载较轻但对显存带宽和 KV 缓存访问频繁（**内存密集型**）。两阶段的区别在于：**Prefill** 一次性处理所有输入，计算密集且并行度高；**Decode** 按 Token 逐步生成，串行依赖之前结果，主要受内存带宽制约。

```python
# 伪代码示例：LLM 推理流程
kv_cache = []
# Prefill 阶段：一次前向计算所有输入token
kv_cache = model.prefill(input_tokens)  # 返回整个输入的KV缓存
# Decode 阶段：自回归生成新token
output = []
current_seq = input_tokens.copy()
while not finished:
    next_token, new_kv = model.decode_step(current_seq, kv_cache)
    output.append(next_token)
    kv_cache.append(new_kv)  # 更新KV缓存
    current_seq.append(next_token)
```

## 为什么要 Prefill–Decode 分离

在传统部署中，Prefill 和 Decode 通常在同一 GPU 上交替执行，这会导致资源浪费和性能瓶颈。首先，两阶段计算特性截然不同：Prefill 是高度并行的矩阵运算，容易饱和 GPU 算力；而 Decode 是低吞吐量的迭代运算，主要受内存带宽限制。当两者混合在一张卡上执行时，一方面 Prefill 执行时 GPU 会被充分利用，但 Decode 任务却往往需要等待；另一方面，Decode 执行时 GPU 大量算力闲置。因此将它们分离在不同硬件上执行，可**匹配专用资源**：给 Prefill 分配高算力节点、给 Decode 分配高带宽显存节点。此外，分离后可**分别优化延迟指标**：例如单独调整 Prefill 阶段的张量并行度以加速首次响应（TTFT），而对 Decode 阶段则用流水线或数据并行提升吞吐或降低逐Token延迟。

从系统调度角度看，分离也可降低不同请求阶段间的**干扰**。传统“连续批处理”（continuous batching）会将 Prefill 和 Decode 请求一起排队批处理，这种做法虽然提高了总体吞吐量，但会导致长请求的 Decode 频繁被插入的 Prefill 打断，增加尾延迟。例如图中示意所示，当系统同时到达多条请求时，Prefill 和 Decode 交错执行会让正在 Decode 的任务频繁停顿，造成不必要的延迟。将两阶段任务拆开后，不同阶段的资源隔离，更易满足不同阶段的延迟/吞吐需求。

## 主流框架中的 Prefill–Decode 分离实现

* **vLLM**：vLLM 是一个高性能企业级推理框架。它默认使用**连续批处理**策略进行Prefill/Decode混合调度（iteration-level scheduling），并引入了**Chunked Prefill**功能：将一个较长的Prefill任务切分成多个较小的块，与Decode请求一起合并进批，以提升GPU利用率。启用后，调度器优先考虑Decode请求，只有在批次剩余Token预算充裕时才插入Prefill；这样可以同时提高每Token延迟和GPU利用率。vLLM 还提供实验性“分离式”部署：通过启动**两个 vLLM 实例**（一个负责 Prefill，一个负责 Decode），并使用内部的 Connector 和 LookupBuffer 将Prefill生成的 KV 缓存传递给 Decode 实例。这种做法允许在不同节点或进程上分别扩展Prefill/Decode能力（见下图）。

![vLLM 中 Prefill-Decode 分离架构示意（Prefill 实例和 Decode 实例通过缓存管道交互）](BigModel/PDD/vLLM.png)

* **TensorRT-LLM**：NVIDIA 的 TensorRT-LLM 引擎同样支持 Prefill/Decode 分离与优化。它可以执行**动态块Prefill（Chunked Prefill）**：将输入Token序列切分为更小片段，逐块执行Prefill，这样可以让后续的Decode任务更快地开始，从而提高 GPU 并行度和吞吐。NVIDIA官方技术博客指出，使用块Prefill后可以“并行化”Prefill和Decode处理，缓解Prefill成为瓶颈的问题。同时，TensorRT-LLM 支持**弹性Chunk大小**自动调节，无需手动指定最大上下文长度。对于批处理，TensorRT-LLM 采用**就绪批处理（in-flight batching）**，即同时处理多个请求的Prefill和Decode，但这仍会导致Decode被Prefill阻塞。通过块Prefill和合适的批次策略，它可以更好地平衡资源利用与延迟。

![TensorRT-LLM 中 Chunked Prefill 的示意（输入序列被切分块并并行处理，以便Prefill和Decode重叠执行）](BigModel/PDD/CP.png)

* **Text Generation Inference (TGI)**：HuggingFace 的 TGI 框架采用客户端-路由器-引擎结构，其路由器组件负责请求排队与连续动态批处理。新请求进入时，需要先执行 Prefill，再与已有的 Decode 批次合并。TGI 使用类似“等待插入”的策略：当等待队列中请求积累到一定阈值时，会**暂停**正在执行的Decode批次来插入新的Prefill计算。例如，它通过 `waiting_served_ratio` 来决定何时将等待的请求加入当前批次，如果加入会增加当前批次的Token数，就中断当前Decode以做Prefill；否则则让等待请求继续等待。这样保证了系统既能保持高吞吐，又不会让新请求无限期延迟。TGI 的配置参数（如 `MAX_BATCH_PREFILL_TOKENS`, `MAX_BATCH_TOTAL_TOKENS` 等）即用于控制 Prefill/Decode 阶段在一次批处理中的Token预算，从而实现高效的批合并和延迟/吞吐权衡。

* **FasterTransformer**：NVIDIA FasterTransformer 是一个底层推理库，提供高度优化的Transformer层实现，支持张量并行和流水线并行跨多 GPU/节点推理。它本身专注于高效执行模型计算，没有自带完整的推理服务器架构或调度策略。通常，FasterTransformer 可作为引擎集成到 Triton 或 TensorRT-LLM 等框架中。在此类场景下，Prefill/Decode 分离由上层框架管理（例如在多个 GPU 上分别部署Prefill和Decode作业），而 FasterTransformer 则在各自阶段发挥加速作用。

## 分离后的调度优化策略

在 Prefill–Decode 分离之后，需要设计高效的调度机制来管理 KV 缓存和批处理：

* **KV 缓存管理**：Prefill 阶段生成的 KV 缓存是 Decode 阶段的关键状态。一般方法是将其保存在 GPU 显存中供同一请求的后续Decode使用。分离架构下，还需在 Prefill 与 Decode 之间传输这些缓存数据。例如 vLLM 利用内存管道（Pipe）或自定义 Connector 在进程间传输 KV；DistServe 通过高速网络传输KV；Dynamo 等系统引入专门的中间队列（PrefillQueue）来缓冲和分发 KV 缓存。此外，还有**自动前缀缓存**（Prefix Caching）机制：如果不同请求具有公共前缀，可在服务端复用已有的 KV 缓存而无需重新计算前缀。vLLM 提供此功能，针对多轮会话或共享提示加速推理。

* **批次合并（Continuous Batching）**：绝大多数框架仍沿用连续批处理策略，将多个请求尽可能合并进一个批次执行，以提高吞吐量。只不过分离后，可更加灵活地“拼接”不同阶段的请求。例如，可以把多个 Decode 请求与一小部分 Prefill 一起批处理，利用计算能力。vLLM 的 Chunked Prefill 就是这种做法：将一个长输入拆块，与其他等待中的 Decode 请求混批执行。TGI 则通过“等待-运行比率”参数动态决定何时暂停Decode去插入Prefill，以尽量利用空闲显存并避免长时间空等。此外，系统通常会限制每次批处理的最大 Token 数（`max_batch_total_tokens`），以防 OOM，并在批次之间**合并等待队列**：当新请求到来但当前批次尚有剩余容量时，它们会被尽快加入以充分利用资源。

* **多路复用与并行**：在跨卡部署时，可配置多个 Prefill 实例和多个 Decode 实例并行运行。比如 Dynamo 可分别启动 N 个 Prefill Worker 和 M 个 Decode Worker，通过路由器平衡负载。这样，GPU 闲置时 Prefill 实例可抢占资源运行，负载波动时可以动态扩缩容。此外，为降低 KV 传输开销，调度器会尽量将某请求的 Prefill 与对应 Decode 安排在网络或同节点带宽较高的设备组内。

## 性能影响

Prefill–Decode 分离在吞吐量、延迟和资源利用上有多方面影响：

* **吞吐量**：单纯拆分并不必然提高每卡原始吞吐（尤其在资源固定的情况下）。例如 vLLM 官方说明指出，“分离式 Prefill 并不提升吞吐”，而研究原型在增加额外 GPU 后才显现吞吐优势。然而，分离让系统可以灵活分配资源（如多张卡用于 Prefill，多张卡用于 Decode），从而在整体资源充足时显著提升有效吞吐。例如 DistServe 实验表明，将 2 个 Prefill 卡配合 1 个 Decode 卡后，系统**人均**吞吐率实现了近 2 倍增长。

* **延迟**：分离可以显著优化**尾延迟和交互延迟**。由于 Prefill 和 Decode 互不干扰，Decode 阶段不再因为突然插入的 Prefill 请求而停顿。而且，分离后可独立调整 Prefill 阶段的拓扑（如张量并行）以缩短首个Token生成时间（TTFT），也可针对 Decode 调度策略降低逐Token延迟（ITL）。NVIDIA 也指出，分块 Prefill 能缩短用户等待下一个Token的时间（减少延迟）。总体来说，分离能提升系统响应一致性，但需要付出一些开销。

* **资源使用**：分离引入了额外开销。例如，需要多分配GPU运行多个实例、额外存储和传输 KV 缓存。但同时也更好地**匹配硬件特点**：Prefill 可使用更激进的低精度算力与大批次并行计算；Decode 可使用带宽更高的显存卡。实测表明，当有高速互连（NVLink/PCIe5.0）时，Prefill 产生的 KV 缓存通过网络传输的延迟可以低于一次 Decode 所需的计算时长，因此对延迟影响微乎其微。

## 挑战与未来方向

尽管 Prefill–Decode 分离带来诸多优势，但也面临挑战：**KV 缓存传输开销**是主要问题之一，尤其在跨节点部署时。研究发现，通过合理分配 Prefill/Decode 节点并利用高速互连，可将该开销降到很低；未来可继续优化，如利用 RDMA、GPUDirect 或更高带宽硬件来减少延迟。另一个挑战是**负载均衡**：需要算法动态决定在何时扩大 Prefill 实例或 Decode 实例的规模，以及如何分配资源以避免一方过载；这涉及系统级调度与预测技术。**推理稳定性和容错**也需注意：当一个阶段节点失败时，如何快速恢复对应缓存成为问题。有研究提出 KV 流式化和缓存冗余等方案来应对。

未来优化方向包括：更细粒度的**动态切分策略**（如根据实际负载自动调整块大小）、更智能的**调度算法**（利用学习或反馈控制自动平衡吞吐与延迟）、以及**多层级缓存**（如在主机内存或SSD中存放部分历史KV，减轻显存压力）。此外，与诸如**投机解码**、**主动剔除低优先级请求**等技术结合也将成为趋势。随着分布式 LLM 推理框架（如 SplitWise、TetriInfer 等）相继采用这一范式，我们预计 Prefill–Decode 分离将逐渐成为大规模 LLM 服务的主流架构。
