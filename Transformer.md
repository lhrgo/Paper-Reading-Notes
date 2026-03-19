
## 前言

今天来看一下Transformer模型，由Google团队提出，论文名为《Attention Is All You Need》。[论文地址](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)。
正如标题所说的，注意力是你所需要的一切，该模型摒弃了传统的RNN和CNN结构，网络结构几乎由Attention机制构成，该论文的亮点在于提出了**Multi-head attention**机制，其又包含了**self-attention**，接下来我们将慢慢介绍该模型的原理。

## 模型架构

![](https://img-blog.csdnimg.cn/2f8c59595bb540ddb21f0d5ea081f312.png#pic_center)

正如文中提到大多数的序列传导模型都含有encoder-decoder结构，Transformer的encoder是将一段表征序列 $(x_1,\cdots,x_n)$ 映射为另一种连续表示的序列 $(z_1,\cdots,z_n)$，即encoder的输出信息；而decoder是将encoder输出和decoder前一步的输出**自回归**的共同生成序列 $(y_1,\cdots,y_m)$。举个例子，现在有一个机器翻译任务，首先将句子embedding为高维向量，输入encoder中，其输出随后输入decoder进行解码得到最终翻译结果，如下图所示。
![Encoder-Decoder](https://img-blog.csdnimg.cn/f8ce9677197d4d7e8f2dd0f7f2ad5da1.png#pic_center)

需要注意的是，Transformer的输出 $y_i$ 是一次一次自回归的生成的，也就是每一次输出都需要调用最后一层encoder的输出序列。这里不像多层RNN隐层的并行传递，Transformer是串行的。

### Encoder和Decoder
好了，接下来该介绍encoder和decoder的神秘面纱了，如下图所示。
![Transformer模型架构](https://img-blog.csdnimg.cn/ba2dc8c48304480a9613d25c758d55ce.png#pic_center)
在读论文时第一眼看这个架构图，一开始是比较懵的，这到底做了些啥操作。后来看了[李沐老师讲解的Transformer](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.337.search-card.all.click&vd_source=ca5cb380410734fb2e1357fe1e104471)才有了一定的理解。

#### Encoder
回到论文的讲解！
![Enc](./assets/transformer/transformer_2.jpg)

这里说到作者实验用到了6层的encoder（串行），这里是为了学到更多的语义信息。并且每层encoder都包含两个子层，分别是**多头注意力机制Multi-head attention**和**前馈神经网络FFN**。当然了，作者对两个子层的输出都做了**residual连接**和**Layer normalization(LN)**，加了残差连接是为了网络能搭的更深，并且容易训练，防止梯度消失；而LN完全是针对**每一个样本**自身的特征缩放，能将每个词都归一为相同空间的语义信息。BN也是一种常见的特征缩放方法，常用于CNN，不适用于NLP任务，因为其对所有batch的同一个特征做缩放，在图像中是非常友好的，而NLP中每一个sequence的长度是不一样的，所以在同一个batch中越长的语句得不到充分的缩放表示。

#### Decoder
![](./assets/transformer/transformer_3.png)

同样的，作者实验用到了6层decoder（串行），不同于encoder，这里作者还设置了**mask**的multi-head attention，其原因在于在解码时，模型是看不到整条句子的，因此，必须在当前时刻掩码掉后面的词（**只能往前看，不能往后看**），才能做到正确训练和有效预测。

### Attention
谈到注意力机制，像我们人一样，看到一幅图片，我们会关注其强烈的表征现象，能让我们快速了解新事物的信息，如下图所示。特别在处理NLP任务中，长距离的记忆能力是一个难题，引入注意力机制，关注更重要的词，可以缓解这一现象。
![在这里插入图片描述](./assets/transformer/transformer_4.png)

在Transformer中，每个单词embedding为三个不同的向量，分别是 $Query$ 向量 $Q$、$Key$向量 $K$ 和 $Value$ 向量 $V$。具体来说，对于一个句子，只需要将其输入到三个linear层，通过学习三个 $W$ 参数就能得到不一样的 $Q、K、V$。至于为什么说 $Q、K、V$ 要不一样，其实一样也可以，但是这里为了**增强数据的表达能力**，保证在 $QK^T$ 矩阵内积时可以**得到不同的空间投影**，**提高模型泛化能力**。

生成的 $Q、K、V$ 矩阵后便可以进行attention计算了，如下图所示
![在这里插入图片描述](https://img-blog.csdnimg.cn/e3460263addc4f178031d4830281bf68.png#pic_center)

假设有三个矩阵 $Q、K、V$，维度分别为 $(d_q, d_{model})、(d_k, d_{model})、(d_v, d_{model})$，其中 $q=k=v$。
1. 进入的是**Multi-Head Attention**多头注意力机制。这里可以h层，也就是我们说的`多头`，类似cv中的channel数量，能学习更多维度信息。多头注意力机制中包含了Scaled Dot-product Attention，也是**self-attention**。
2. 进入self-attention，对于每个sequence，用它的query矩阵: $(d_q, d_{model})$ 和key向量shape: $(d_k, d_{model})$进行内积，本质上是求解每个词之间的`余弦相似度`，如果两者相似度较高，则赋予较大的值来反映两者的关系，反之如果是正交的，内积为0，则它们就没有相似性，这里输出的attention score矩阵维度是shape: $(d_q, d_k)$。
3. 将输出矩阵进行scale缩放相似度，因为 $d_{model}$ 一般会比较大，导致矩阵数值大小不一，使得softmax容易推向饱和区01，使得收敛困难，公式如 $\text{Attention}(Q、K、V)$ 所示。
4. 通过可选的mask操作，为了保证decoder得不到sequence的leak信息。具体来说是通过将权重矩阵 $QK^T$ 上三角改为负无穷，这样softmax就能将这些值推为0，即无权重，保证mask的作用。
5. 将attention score矩阵shape: $(d_q, d_{k})$ 与Value矩阵shape: $(d_v, d_{model})$ 内积，得到encoder后的sequence信息表征shape: $(d_q, d_{model})$。

Scale缩放公式：

$$
\begin{aligned}
    \text{Attention}(Q、K、V)=\text{softmax}(\frac{QK^T}{\sqrt(d_k)})V
\end{aligned}
$$

Multi-head公式：

$$
\begin{aligned}
    \text{Multihead}(Q、K、V)=\text{Concat}(\text{head}_i,\cdots,\text{head}_h)W^O\\
    \text{where} \text{ } \text{head}_i=\text{Attention}(QW_{i}^{Q},QW_{i}^{K},QW_{i}^{V})
\end{aligned}
$$

上述的操作执行完后，便可以通过多个头的concat将矩阵拼接，随后通过linear层降维，完成Multi-head attention的过程。

![](https://img-blog.csdnimg.cn/ef07b6f233874ffc84623d1ed81da455.png#pic_center)
值得注意的是多头数量必须可被$d_{model}$整除。这个很好理解，在CNN中，我们经常将feature map的width和height降维后，会把channel数升高，学到更深的信息一个道理。

### FFN
除了注意子层外，encoder和decoder中的每个层都包含一个完全连接的前馈网络，它分别和相同地应用于**每个位置**（**逐token应用，画重点！！**）。这由两个线性变换组成，中间有一个ReLU激活。换句话说就是MLP模型。公式如下所示：

$$
\begin{aligned}
    \text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2
\end{aligned}
$$

![](https://img-blog.csdnimg.cn/c2517bc087864d8ca5d306e0428a52b6.png#pic_center)
论文作者给定了MLP的中间状态输出维度为2048，而最后输出维度为512，当然就是512->2048->512这样变换。

### Embeddings和Positional Encoding
![Emdedding+Positional Encoding](https://img-blog.csdnimg.cn/e9479a23363447bb80aaee65e56757b2.png#pic_center)

#### Embeddings
在Transformer中，在嵌入层中，这些权重会乘 $\sqrt{d_{model}}$ **（一个小细节）**。其原因是在嵌入层学emdedding的时候，权重都会比较小，但后续要和positional encoding相加，需要保持差不多的scale。

#### Positional Encoding
self-attention对输入sequence中各单词的位置或者说顺序不敏感，因为通过Query向量和Key向量的内积，本质上就是一些词由其他词的线性表出，并没有说有位置的信息存在。比如“我吃牛肉”这句话，在Transformer看来和“牛吃我肉”是没什么区别的。
为了缓解该问题，作者提出了位置编码。简而言之，就是在词向量输入到注意力模块之前，与该词向量等长的位置向量进行了按位相加，进而赋予了词向量相应的位置信息。

作者给出了位置编码的定义公式，具体如下：

$$
PE_{pos, 2i}=sin(pos/10000^{2i/d_{model}}) \\
PE_{pos, 2i+1}=cos(pos/10000^{2i/d_{model}})
$$

这样通过 $sin(\alpha+\beta)=sin(\alpha)cos(\beta)+cos(\alpha)sin(\beta)$ 。可以将牛(pos=3)可以由pos=2和pos=4表达，使得Transformer可以更容易掌握单词间的相对位置。
## 例子
举个例子，现在有一个任务，根据历史24个点 $(x_1,\cdots, x_{24})$ 预测未来48个点 $(x_{25},\cdots, x_{72})$。

一个样本对：
$X: (batch=32, seq=24, fea=1)$
$Y_{in}: (batch=32, seq=48, fea=1)$
$Y_{out}: (batch=32, seq=48, fea=1)$

$X$ 是 $x_1,\cdots, x_{24}$ 用于Encoder输入
$Y_{in}$ 是 $x_{24},\cdots, x_{71}$ 用于Decoder输入
$Y_{out}$ 是 $x_{25},\cdots, x_{72}$ 用于计算Loss

**Encoder(train)**
1. Input Embedding + Positional Encoding --> (32, 24, 512) + (24, 512) = (32, 24, 512) == Q=K=V
2. Multi-Head Attention
- Q、K、V 分别Linear --> (32, 24, 512)
- Q、K、V 分别reshape+transform --> (32, 8, 24, 64)
- 并行h=8个Self-Attention --> (32, 8, 24, 64)
- Concat --> (32, 24, 512)
- Linear --> (32, 24, 512)
3. Add & Norm --> (32, 24, 512)
4. FFN(单token) --> (32, 24, 2048) --> (32, 24, 512)
5. Add & Norm --> (32, 24, 512)

``串行重复步骤2、3、4、5共6次`` --> (32, 24, 512) == K=V

**Decoder(train)**
1. Output Embedding + Positional Encoding --> (32, 48, 512) + (48, 512) = (32, 48, 512) == Q=K=V
2. Masked Multi-Head Attention --> (32, 48, 512)
3. Add & Norm --> (32, 48, 512)
4. Encoder-Decoder Multi-Head Attention --> (32, 48, 512)
5. Add & Norm --> (32, 48, 512)
6. FFN(单token) --> (32, 48, 2048) --> (32, 48, 512)
7. Add & Norm --> (32, 48, 512)

``串行重复步骤2、3、4、5、6、7共6次`` --> (32, 48, 512)

8. Linear --> (32, 48, 1)

计算Loss，反向传播，重复循环。

**Encoder(infer)**
使用最后一个样本(1, 24, 1) --> (1, 24, 512) == K=V ``(仅生成一次)``
**Decoder(infer)**
使用最后一个样本最后一时刻(1, 1, 1)== $y_{0}$ --> (1, 1, 1)==$\hat{y}_{1}$

拼接 $y_{0}, \hat{y_{1}}$=(1, 2, 1) --> (1, 2, 1)，取最后一刻即(1, 1, 1)== $\hat{y}_{2}$

拼接 $y_{0}, \hat{y_{1}}, \hat{y_{2}}$=(1, 3, 1) --> (1, 3, 1)，取最后一刻即(1, 1, 1)== $\hat{y}_{3}$

重复直到生成 $\hat{y}_{1}, \cdots \hat{y}_{48}$ **（注意Mask Attention部分可以缓存 $K_{old}$ 和 $V_{old}$，下一步直接计算新pre值的K和V，分别拼接 $K_{old}$ 和 $V_{old}$，能减少许多不必要的计算。）**

## 总结

说一下计算效率：
- 串行计算：Encoder和Decoder分别有6层，这都需要把上一层输出来输入到下一层。
- 并行计算：多头注意力这里有8层，完全能并行。训练时由于decoder是输入完整的seq计算self-att，所以也能并行。

关于Transformer比较重要的点基本上就这些，当然还有很多细节的地方需要去探索，接下来我将会写更多的论文分享，总结一些经典的模型。