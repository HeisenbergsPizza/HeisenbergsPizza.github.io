---
layout: post
title: "Transformer"
date: 2025-08-13 01:19:00 +0800
categories: [AI]
tags: [Transformer]
---
transformer最初被用于文本的翻译任务，因此从这里开始

由于输入形式与输出形式都是words sequence，因此首先需要对输入sequence进行的操作是Word Embedding，即将每一个word都使用维度为 $d$ 的向量表示，我们假设输入的序列长度为 $L$，那么我们就可以将这个句子表示为一个 $L\times d$ 的矩阵。但是这还不够，此时我们还无法表示每一个word的位置信息，因此还需要计算positional encoding。positional encoding由固定公式计算得到，与位置上的具体word无关，因此对于每个长度一样的句子来说都加上一个相同的positional encoding矩阵 $a$。相加即得到输入矩阵 $\mathbf{X}^{L\times d}$，将这个矩阵输入self-attention。

## Self-Attention Mechanism

用可学习（训练过程确定）的线性变换将输入矩阵中的每个word对应的向量映射为Query、Key、Value：

$$
Q=XW^{Q}+b^{Q}, K=XW^{K}+b^{K},V=XW^{V}+b^{K}
$$

单头注意力中，$X^{Q},X^{K},X^{V}\in \mathbb{R}^{d\times d}$，因此 $Q,K,V \in \mathbb{R}^{L\times d}$，将每个query与所有keys做点积得到打分矩阵：

$$
scores=\frac{QK^{T}}{\sqrt{d_{k}}} \in \mathbb{R}^{L\times L}
$$

$d_{k}$ 为单头维度（single-head时 $d_{k}=d$），是为了数值稳定和梯度稳定。

**Q与K相乘含义解释**：在训练好的嵌入空间中，相关性高的words所对应的向量相似度也比较高，相应的点积得到的结果也就越大。score(i,j)是由Q的第i行乘以K的转置前的第j行，而 $q_{i}=x_{i}W^{Q}$，$k_{j}=x_{j}W^{K}$，在$W^{Q},W^{K}$训练好的情况下，相乘得到的结果即与第i个word和第j个word之间的相似程度成正相关。

接下来对每一个列向量做 softmax 得到注意力权重：$A=\mathbf{softmax}(scores)$，即对第i个word与其他所有word的相似程度进行归一化。

此时将权重与values相乘得到每个query的输出：$\mathbf{Attention}(Q,K,V)=AV \in \mathbb{R}^{L\times d_{k}}$，可以理解为每个输出位置都对序列所有位置value做加权平均，权重由query与各key的相似程度确定。

## Multi-Head Attention

单头注意力的 $Q,K,V$ 都是原始的embedding经过同一组线性变换得到的，可能导致理解能力单一、只关注到一种模式的相似。

多头注意力是在经过线性变换得到 $Q,K,V$ 后，将这三个矩阵分别在模型维度上split为 h 个（原论文为8）矩阵，从而得到 $Q_{heads},K_{heads},V_{heads}\in \mathbb{R}^{h\times L\times \frac{d}{h}}$

对于第i个头，$\mathbf{Attention}(Q_{i},K_{i},V_{i})=\mathbf{softmax}(\frac{Q_{i}K_{i}^{T}}{\sqrt{d/h}})V_{i} \in \mathbb{R}^{h\times L\times \frac{d}{h}}$

将所有头的结果在划分的模型维度上拼接起来：$Concat(head_{1},...,head_{h})\in \mathbb{R}^{L\times d}$

由于直接拼接起来的结果每一段都是独立的表示，因此拼接起来还需要经过一个线性变换层来将不同头捕获到的特征融合成一个整体表示。
