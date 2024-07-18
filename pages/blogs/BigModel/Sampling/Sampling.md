![Sampling](BigModel/Sampling/Sampling.png)
## 大模型中的采样选择机制详解

在自然语言处理（NLP）和生成模型（如GPT）中，采样选择机制是一种从模型的概率分布中选择词的方法，用于控制生成文本的多样性和质量。本文将详细介绍几种经典的采样选择机制，包括随机采样、Top-k采样、Top-p（Nucleus）采样、温度采样、束搜索（Beam Search）和逆向温度采样，并配以公式和代码示例。

### 一、采样选择机制概述

采样选择机制通过不同的方法从模型的输出概率分布中选择下一个生成的词，从而影响生成文本的特性和质量。

### 二、经典采样选择机制

#### 1. 随机采样

**随机采样**是最简单的一种方法，直接从模型输出的概率分布中随机选择下一个词。它保留了概率分布的多样性，但可能生成不连贯的文本。

**公式描述**：

给定词汇表`V`和概率分布`P_t`，在时间步$t$时，根据概率分布直接进行采样：

$$
w_t \sim P_t(w)
$$

**代码示例**：

```python
import torch
import torch.nn.functional as F

def random_sampling(logits):
    """
    随机采样
    :param logits: 模型输出的logits
    :return: 采样得到的下一个词的索引
    """
    # 计算概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 根据概率分布进行采样
    next_token = torch.multinomial(probs, 1)
    
    return next_token.item()

# 示例logits
logits = torch.tensor([2.5, 1.2, 0.3, 3.7, 0.8])

# 执行随机采样
next_token_index = random_sampling(logits)
print("随机采样得到的下一个词索引：", next_token_index)
```

#### 2. Top-k采样

**Top-k采样**通过选择概率最高的k个词，截断概率分布以限制候选集，随后从中采样。这种方法可以减少生成不合理词的概率。

**公式描述**：

1. 对概率分布`P_t`进行排序，得到前k个最高概率的词`w_{i_1}, w_{i_2}, ..., w_{i_k}`及其对应的概率`P_t(w_{i_1}), P_t(w_{i_2}), ..., P_t(w_{i_k})`。
2. 截断并重新归一化：
    $$
    \hat{P_t}(w_{i_j}) = \frac{P_t(w_{i_j})}{\sum_{j=1}^{k} P_t(w_{i_j})}
    $$
3. 根据重新归一化后的概率分布进行采样。

**代码示例**：

```python
import torch
import torch.nn.functional as F

def top_k_sampling(logits, k):
    """
    根据给定的logits进行Top-k采样
    :param logits: 模型输出的logits
    :param k: Top-k值
    :return: 采样得到的下一个词的索引
    """
    # 对logits进行排序并截断
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # 重新归一化概率
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    
    # 根据概率分布进行采样
    next_token = torch.multinomial(top_k_probs, 1)
    
    # 获取对应的词汇索引
    next_token_index = top_k_indices[next_token]
    
    return next_token_index.item()

# 示例logits
logits = torch.tensor([2.5, 1.2, 0.3, 3.7, 0.8])

# 执行Top-k采样
next_token_index = top_k_sampling(logits, k=3)
print("Top-k采样得到的下一个词索引：", next_token_index)
```

#### 3. Top-p（Nucleus）采样

**Top-p（Nucleus）采样**通过选择累计概率达到某个阈值p的最小词集，动态调整候选集的大小，从而在控制多样性和质量之间取得平衡。

**公式描述**：

1. 对概率分布`P_t`进行排序，得到排序后的词集合`w_1, w_2, ..., w_V`及其对应的概率`P_t(w_1), P_t(w_2), ..., P_t(w_V)`。
2. 选择最小的词集合使得累计概率达到阈值p：
    $$
    \sum_{i=1}^{m} P_t(w_i) \geq p
    $$
3. 截断并重新归一化选择的词集合的概率。
4. 根据重新归一化后的概率分布进行采样。

**代码示例**：

```python
import torch
import torch.nn.functional as F

def top_p_sampling(logits, p):
    """
    根据给定的logits进行Top-p采样
    :param logits: 模型输出的logits
    :param p: Top-p值
    :return: 采样得到的下一个词的索引
    """
    # 计算概率分布并排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # 计算累计概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累计概率大于p的最小索引
    cutoff_index = torch.where(cumulative_probs >= p)[0][0]
    
    # 截断并重新归一化
    top_p_probs = sorted_probs[:cutoff_index + 1]
    top_p_indices = sorted_indices[:cutoff_index + 1]
    top_p_probs /= top_p_probs.sum()
    
    # 根据概率分布进行采样
    next_token = torch.multinomial(top_p_probs, 1)
    
    # 获取对应的词汇索引
    next_token_index = top_p_indices[next_token]
    
    return next_token_index.item()

# 示例logits
logits = torch.tensor([2.5, 1.2, 0.3, 3.7, 0.8])

# 执行Top-p采样
next_token_index = top_p_sampling(logits, p=0.8)
print("Top-p采样得到的下一个词索引：", next_token_index)
```

#### 4. 温度采样

**温度采样**通过调整概率分布的“温度”参数来控制生成文本的多样性。温度越高，生成的文本越多样化；温度越低，生成的文本越确定性。

**公式描述**：

给定词汇表`V`和概率分布`P_t`，在时间步$t$时，通过调整温度参数$\tau$得到新的概率分布：

$$
P_t(w_i) = \frac{\exp(\frac{logits(w_i)}{\tau})}{\sum_{j=1}^{V} \exp(\frac{logits(w_j)}{\tau})}
$$

其中，$\tau$为温度参数。

**代码示例**：

```python
import torch
import torch.nn.functional as F

def temperature_sampling(logits, temperature=1.0):
    """
    温度采样
    :param logits: 模型输出的logits
    :param temperature: 温度参数
    :return: 采样得到的下一个词的索引
    """
    # 调整logits的温度
    adjusted_logits = logits / temperature
    
    # 计算概率分布
    probs = F.softmax(adjusted_logits, dim=-1)
    
    # 根据概率分布进行采样
    next_token = torch.multinomial(probs, 1)
    
    return next_token.item()

# 示例logits
logits = torch.tensor([2.5, 1.2, 0.3, 3.7, 0.8])

# 执行温度采样
next_token_index = temperature_sampling(logits, temperature=0.7)
print("温度采样得到的下一个词索引：", next_token_index)
```

#### 5. 束搜索（Beam Search）

**束搜索（Beam Search）**是一种启发式搜索算法，通过保留多个候选序列来寻找最优序列。束搜索在每个时间步保留固定数量的候选序列，并扩展这些候选序列直到达到最大长度。

**公式描述**：

1. 初始化`beam_width`个候选序列，每个序列的初始概率为1。
2. 在每个时间步，扩展每个候选序列，生成新的候选序列。
3. 对所有新的候选序列进行排序，保留前`beam_width`个最优序列。
4. 重复步骤2和3，直到达到最大序列长度或

满足终止条件。

**代码示例**：

```python
import torch
import torch.nn.functional as F

def beam_search(logits_fn, initial_input, beam_width=3, max_length=20):
    """
    束搜索
    :param logits_fn: 生成下一个词的logits函数
    :param initial_input: 初始输入
    :param beam_width: 束宽度
    :param max_length: 最大序列长度
    :return: 最优序列
    """
    sequences = [[initial_input, 1.0]]
    
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            logits = logits_fn(seq)
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, beam_width)
            
            for i in range(beam_width):
                candidate = [seq + [top_k_indices[i].item()], score * top_k_probs[i].item()]
                all_candidates.append(candidate)
        
        # 对所有候选序列进行排序，保留前beam_width个最优序列
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]
    
    return sequences[0][0]

# 示例logits函数
def example_logits_fn(seq):
    # 简单模拟logits输出
    return torch.tensor([2.5, 1.2, 0.3, 3.7, 0.8])

# 执行束搜索
initial_input = [0]
best_sequence = beam_search(example_logits_fn, initial_input, beam_width=3, max_length=5)
print("束搜索得到的最优序列：", best_sequence)
```

#### 6. 逆向温度采样（Reverse Temperature Sampling）

**逆向温度采样**通过逐步提高温度参数，从确定性较高的分布逐步过渡到多样性更高的分布。这种方法可以生成更加自然的文本。

**公式描述**：

给定初始温度$\tau_0$和增长速率$\alpha$，在每个时间步$t$调整温度参数：

$$
\tau_t = \tau_0 \cdot \alpha^t
$$

**代码示例**：

```python
import torch
import torch.nn.functional as F

def reverse_temperature_sampling(logits, initial_temperature=1.0, alpha=1.1, step=0):
    """
    逆向温度采样
    :param logits: 模型输出的logits
    :param initial_temperature: 初始温度
    :param alpha: 温度增长速率
    :param step: 当前时间步
    :return: 采样得到的下一个词的索引
    """
    # 计算当前时间步的温度
    temperature = initial_temperature * (alpha ** step)
    
    # 调整logits的温度
    adjusted_logits = logits / temperature
    
    # 计算概率分布
    probs = F.softmax(adjusted_logits, dim=-1)
    
    # 根据概率分布进行采样
    next_token = torch.multinomial(probs, 1)
    
    return next_token.item()

# 示例logits
logits = torch.tensor([2.5, 1.2, 0.3, 3.7, 0.8])

# 执行逆向温度采样
next_token_index = reverse_temperature_sampling(logits, initial_temperature=1.0, alpha=1.1, step=2)
print("逆向温度采样得到的下一个词索引：", next_token_index)
```

### 三、总结

本文详细介绍了大模型中的几种经典采样选择机制，包括随机采样、Top-k采样、Top-p（Nucleus）采样、温度采样、束搜索（Beam Search）和逆向温度采样。每种机制有不同的特点和适用场景，选择适当的机制可以有效地控制生成文本的质量和多样性。希望通过本文的介绍，读者能够理解并应用这些采样选择机制，提高生成模型的表现。