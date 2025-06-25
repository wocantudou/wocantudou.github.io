![CP](BigModel/CP/CP.png)
# 什么是Chunked Prefill技术？

## 基本定义与适用场景

 大模型生成推理一般包含两个阶段：**Prefill（预填充）** 阶段对输入提示（prompt）进行前向推理，构建 Key-Value (KV) 缓存；**Decode（解码）** 阶段逐步生成输出令牌。传统流程中，Prefill 会处理整个输入序列后才开始输出解码，此时并行度高但对后续解码造成阻塞。**Chunked Prefill（分块预填充）** 即将一个长prompt的Prefill过程拆分成多个小块（chunk），在调度时将这些预填充块与其他请求的解码任务混合批处理。当多个请求并发到达（如多用户并行查询、服务器高并发负载）或在多轮对话中上下文不断累积时，Chunked Prefill 可以避免让其他请求的解码阶段因一个长Prompt的完整Prefill而完全停顿，提升吞吐并降低平均延迟。换言之，Chunked Prefill 在多请求并发推理场景下，将Prefill拆分并优先执行解码任务，从而**提升并行度和GPU利用率**，同时提供更多机会让解码阶段“搭载”在预填充计算上执行。

## 工作机制与技术原理

由于Prefill处理整个输入一次性执行，属于**计算密集型**操作；而Decode每次只处理一个令牌，计算开销小但受限于内存带宽，是**内存带宽敏感型**操作。Chunked Prefill 的核心思路是**交错执行计算密集型和内存密集型任务**：将长Prompt的Prefill拆成若干大小相等的子块（如每块512或1024个token），然后在构造执行批次时，每个批次只包括一个Prefill块和尽可能多的解码任务。这样，一个Prefill块就能充分利用GPU的计算资源，而其他解码请求则“搭车”加入同一个批次，以较小的增量成本完成计算。经过一个块的前向计算后，GPU上生成的部分KV缓存即被更新（相当于完成这部分Prompt的处理），然后进行下一块的计算；如此迭代直到整个Prompt处理完毕。关键是**正确设置Attention Mask**，使每一块计算与完整Prefill等价：每个子块只关注此前已处理的上下文，确保分块计算结果与一次性完整Prefill相同。

在调度层面，启用Chunked Prefill后，推理引擎通常会**优先调度解码请求**，然后利用剩余“令牌预算”（token budget）安排待处理的Prefill请求。如果一个大Prefill无法完全放入当前批次，则将其拆成更小的块。例如，vLLM的调度策略就是：首先批处理所有等待的Decode请求，然后在剩余空间中安排Pending的Prefill请求，当遇到无法完整放入的Prefill，就将其分割为多个块。NVIDIA的TensorRT-LLM同样支持将令牌分成小块预填充，从而“防止预填充成为瓶颈”，并使解码阶段与之并行，提高整体GPU利用率。总体而言，Chunked Prefill通过**将本来串行的Prefill和Decode阶段并行化**，在逻辑上实现了多请求（或单请求的多阶段）之间的工作负载重叠，从而提高计算利用率、平滑资源占用曲线并减少流水线停顿（bubbles）。

## 实现流程与框架案例

典型的Chunked Prefill实现大致流程如下：

1. **接收请求并初始化**：每个输入请求包含Prompt（可能很长）和解码所需的上下文缓存。系统为每个请求维护已完成的Prefill块数和当前的KV缓存状态。
2. **拆分Prefill块**：对于长度超过预设块大小的Prompt，系统计算需要拆分成多少个块。例如，将1000 tokens拆为4块，每块250 tokens，并设置相应的Attention Mask，保证每块只关注已完成的上下文。
3. **调度批次构建**：调度器在每次迭代时优先选取等待的Decode任务填满批次，然后在剩余位置安排一个Pending的Prefill块（或多个较小块，取决于`max_num_batched_tokens`等配置）。这样，一个批次通常包含1个Prefill块和若干个Decode令牌。
4. **执行前向计算**：对批次同时执行前向推理。若批次含Prefill块，则该块的输出更新对应请求的KV缓存；若含Decode令牌，则输出对应的生成令牌（并更新各自请求的KV缓存）。执行完成后，调度器将更新各请求的进度状态：已完成的Prefill块数增加，或生成的新令牌数增加。
5. **循环迭代**：返回步骤3，继续处理剩余任务。一个请求的Prefill直到最后一块完成后才开始输出最终生成。当所有请求的输入都处理完毕后，仅剩Decode阶段，系统可恢复对解码任务的连续批处理。

在具体框架中的实现要点包括：**vLLM**（开源高效推理引擎）通过设置 `enable_chunked_prefill=True` 开启此功能。其内部调度逻辑在论文：[SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/pdf/2308.16369.pdf)已有详细描述：启用后，调度器会先聚集所有Decode请求，然后考虑Prefill任务，不足以放入的长Prefill会被拆块。同时，vLLM利用PagedAttention技术管理KV缓存的内存分配，使得分块后的缓存无需连续存储，也不浪费内存。**NVIDIA TensorRT-LLM** 引擎同样内置Chunked Prefill支持，并采用动态块大小机制：用户或系统根据GPU利用率自动选择合适的Prefill块大小，既兼顾 **时间到首令牌（TTFT）** 又考虑总体吞吐。TensorRT-LLM通过将长上下文解耦出单次激活缓冲区大小，使得更长Context也能被处理而不额外占用显存。**Hugging Face TGI（Text Generation Inference）** 在其3.3.0版本中也添加了对Chunked Prefill的支持（Release Notes中提及“Chunked Prefill VLM”特性）。TGI结合连续批处理和前缀缓存策略，同样能够在多个请求间交错执行Prefill和Decode来提高并发吞吐。但与vLLM类似，具体调度逻辑保持“优先解码、再处理Prefill块”的原则，以减少解码等待。

## 性能优化效果

Chunked Prefill在多项研究和测试中均展现了显著的性能提升。总体来说，其**吞吐量和延迟**表现都有优化：首先，由于解码阶段不再长时间空闲，整体吞吐率大幅提高。Sarathi论文实验显示，在A6000 GPU上运行LLaMA-13B模型时，Chunked Prefill将Decode吞吐率提升了**10倍**，端到端吞吐提升了**33%**；在A100上对LLaMA-33B模型，Decode吞吐提升约**4.25倍**，端到端提升**25%**。vLLM团队在实际自研测试中也观测到显著提升：启用分块预填充后，在请求大小均匀的情况下，总令牌吞吐率提升约**50%**。从延迟角度看，Chunked Prefill通过使Prefill与Decode并行，能够缩短请求的**尾部延迟**和**平均每令牌延迟**。华为团队指出，Chunked Prefill平衡了Prefill和Decode的计算利用率，可降低P90的TTFT（首令牌延迟）和P90的每令牌延迟。vLLM内部数据表明：在持续高负载场景下，Chunked Prefill可使交互式每令牌延迟（ITL）降低约10%–20%，在高并发时端到端吞吐可提高近**2倍**；代价是比默认策略略高的首令牌延迟（TTFT），因为默认调度本来就是为最快得到首令牌而优化的。总的来说，Chunked Prefill在提高GPU利用率、缩短平均生成时间方面效果明显，已成为自托管LLM服务的推荐默认策略之一。
