![ELBO](ML/ELBO/ELBO.png)
# 复习变分下界即证据下界（Evidence Lower Bound, ELBO）：原理与应用

变分下界（**Variational Lower Bound**），也称为“**证据下界**”（**Evidence Lower Bound, ELBO**），是概率模型中的一个重要概念，广泛用于变分推断（**Variational Inference, VI**）等领域。变分推断是一种近似推断方法，它通过将复杂的后验分布用一个易于处理的分布来近似，从而使得计算变得可行。变分下界是推导和优化这个近似分布的核心工具。本文将复习变分下界的基本概念、推导过程及其在机器学习中的应用。

## 1. 概念背景

在贝叶斯推断框架中，我们通常希望根据观测数据 $x$ 来推断潜在变量 $z$ 的后验分布 $p(z|x)$。后验分布的计算通常依赖于边缘似然（Marginal Likelihood）：

$$
p(x) = \int p(x|z) p(z) \, dz
$$

这个积分通常非常复杂，因此直接计算 $p(z|x)$ 并不现实。为了解决这一问题，变分推断通过引入一个近似分布 $q(z)$ 来逼近后验分布 $p(z|x)$，并通过最优化使得 $q(z)$ 尽可能接近 $p(z|x)$。

衡量 $q(z)$ 和 $p(z|x)$ 之间的差异最常用的工具是 **KL 散度**（Kullback-Leibler divergence）：

$$
\text{KL}(q(z) \parallel p(z|x)) = \int q(z) \log \frac{q(z)}{p(z|x)} dz
$$

KL 散度是非负的，并且只有在 $q(z) = p(z|x)$ 时才为零。通过最小化 KL 散度，我们可以找到最优的 $q(z)$。

## 2. 变分下界的推导

为了推导变分下界，我们从边缘似然的对数形式开始：

$$
\log p(x) = \log \int p(x|z) p(z) dz
$$

直接计算这一积分非常困难，因此我们引入 $q(z)$ 并利用 Jensen 不等式，得到：

$$
\log p(x) \geq \int q(z) \log \frac{p(x, z)}{q(z)} dz
$$

右侧的表达式就是变分下界，即 **ELBO**。通过最大化这个下界，我们可以优化 $q(z)$ 使其尽量接近后验分布 $p(z|x)$。

进一步地，变分下界可以拆分为以下两部分：

$$
\mathcal{L}(q) = \mathbb{E}_{q(z)}[\log p(x|z)] - \text{KL}(q(z) \parallel p(z))
$$

- **第一项** $\mathbb{E}_{q(z)}[\log p(x|z)]$ 是对数似然的期望，衡量模型对观测数据的拟合能力。
- **第二项** $\text{KL}(q(z) \parallel p(z))$ 则衡量近似分布和先验分布之间的距离。

最大化 ELBO 使得近似分布 $q(z)$ 同时具备良好的数据拟合能力，并不会偏离先验分布太多。

## 3. 变分下界的应用

### (1) 变分自编码器（**Variational Autoencoder, VAE**）

VAE 是变分推断在深度学习中的一个重要应用。它使用神经网络来参数化近似分布 $q(z|x)$ 和生成分布 $p(x|z)$，并通过最大化变分下界来训练模型。VAE 的目标函数为：

$$
\mathcal{L}(q) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \parallel p(z))
$$

其中，VAE 利用 **重参数化技巧**（Reparameterization Trick）来确保梯度能够正常反向传播，这一技巧是 VAE 的核心技术之一。

### (2) 无监督学习与 LDA

在主题模型如隐含狄利克雷分配（LDA）中，变分推断用于近似推断文档的主题分布。通过优化变分下界，LDA 能够有效地提取文档的潜在结构，并应用于推荐系统和文本分析等领域。

### (3) 贝叶斯神经网络

贝叶斯神经网络通过引入变分推断来近似神经网络中的权重后验分布，这种方法能够有效量化模型的不确定性，从而增强泛化能力，特别适用于强化学习和决策系统。

## 4. 变分下界的局限性与改进

### 近似分布的限制

变分推断中，选择较为简单的分布（如高斯分布）来近似真实后验，可能无法准确描述复杂的后验分布。为了应对这个问题，基于流的方法（Flow-based Variational Inference）通过引入可逆神经网络增强了近似分布的表现力。

### 局部最优问题

由于优化的非凸性，变分推断容易陷入局部最优。针对这一问题，研究者提出了更多基于采样的方法，如 **Monte Carlo 变分推断**，以提高近似的质量。

## 5. 总结

变分下界是变分推断的核心工具，它通过最大化下界来找到近似的后验分布，使得复杂的推断问题变得可解。尽管变分下界在一些场景中存在局限性，但它仍然是生成模型、贝叶斯方法和无监督学习中的重要组成部分。随着算法的改进，变分推断及其下界的应用将会更加广泛。

## 参考文献

1. Kingma, D.P., Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.
2. Blei, D.M., Kucukelbir, A., McAuliffe, J.D. (2017). Variational Inference: A Review for Statisticians. J. of the American Statistical Association.
