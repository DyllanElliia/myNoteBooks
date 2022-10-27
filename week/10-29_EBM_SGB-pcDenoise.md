# 22-10-29周报

## 论文研读

> 本周学习了Score-Based Model的基础内容EBMs，并对其中基础推导，结合其他资料梳理了一遍笔记。
>
> 完成了对SBG-based的点云降噪进行了研读。
>
> [How to Train Your Energy-Based Models](https://arxiv.org/abs/2101.03288)
>
> [Score-Based Point Cloud Denoising](https://openaccess.thecvf.com/content/ICCV2021/html/Luo_Score-Based_Point_Cloud_Denoising_ICCV_2021_paper.html)

### Score-Based generative Model



EBM是当前score-based方法的基础模型。

Score-Based generative Model（以下简称SBG）有如下优点：

- 无需对抗学习，得到GAN级别的采样效果；
- 灵活的模型结构，基于这类方法可以设计符合课题的模型；
- 精确的exact log-likelihood computation（对数似然估计）；
- uniquely identifiable representation learning；

符号表：

|      Symbol       | Description                      |
| :---------------: | :------------------------------- |
| $x_i,\ i=1,2,...$ | 第$i$个数据集                    |
|     $\theta$      | 模型的参数，训练时需要寻找的参数 |
|   $p_\theta(x)$   | 模型的概率密度函数               |
|   $p_{data}(x)$   | 未知的真实数据的概率密度函数     |
|   $E_\theta(x)$   | Energy function，可以是任意函数  |

#### Energy-based model(EBM)

EBM源于统计动力学，EBM的模型可被描述为如下形式：
$$
p_\theta(x)=\frac{e^{-E_\theta(x)}}{Z_\theta},\\where:Z_\theta=\int_\infty e^{-E_\theta(x)}dx
$$
其中，$Z_\theta$的作用是让概率密度总为`1`；结果上，能量$E_\theta(x)$越低的状态越有可能发生。

EBM是一种likelihood-based的直接学习data-generating分布的方法。但使用EBM建模时，不使用传统损失函数进行计算，以下用negative log-likelihood举例：
$$
\begin{align*}

\sum_i-{\rm{log}}\ p_\theta(x_i)&=\sum_i-{\rm{log}}\frac{e^{-E_\theta(x_i)}}{Z_\theta}\\
&=\sum_iE_\theta(x_i)+\color{BrickRed}{{\rm{log}}\int e^{-E_\theta(x_i)}dx}


\end{align*}
$$
显然，若要用梯度下降计算$\theta$，离不开处理后面的积分，这是个非常大难以实现的计算量。因此SCG方法使用了一个近似方法实现likelihood。

#### Score Matching

从上可知，对$\theta$求导必然要处理$Z_\theta$问题，因此Score-based方法转而对$x$进行求导，定义score function $s(x)$为log-density function对$x$的梯度。其效果如下：
$$
\begin{align*}
s_\theta(x)=\nabla_x{\rm{log}}p_\theta(x)&=-\nabla_xE_\theta(x)-\nabla_x{\rm{log}}Z_\theta\\
&=-\nabla_xE_\theta(x)

\end{align*}
$$
同时，我们希望$p_\theta(x)$尽可能接近目标$p_{data}(x)$，若有$s_\theta(x)=s_{data}(x)$，则：
$$
E_\theta(x)=E_{data}(x)+Constant
$$
又因为：
$$
\begin{align*}
p_\theta(x)&=\frac{e^{-E_\theta(x)}}{Z_\theta}\\
&=\frac{e^{-E_{data}(x)-c}}{\int e^{-E_{data}(x)-c}dx}\\
&=\frac{e^{-E_{data}(x)}}{Z_{data}}\\
&=p_{data}(x)
\end{align*}
$$
因此，score相等等价于分布$p$相等，即寻找参数$\theta$问题可被约化为对二者的score进行比较。由此，我们可以定义新的loss：（这里出现了符号冲突，定义$E[x]$为求集合$x$的期望)
$$
\mathcal L=E_{data}[\{s_\theta(x)-s_{data}(x)\}^2]
$$
这种loss被称为Fisher Divergence。然而，$s_{data}(x)$也是未知项，因为我们不知道数据的**真实**分布。前人研究发现，在一些条件下，该loss可被展开为一个不含$s_{data}(x)$的表达式：
$$
\begin{align*}
E_{data}[\{s_\theta(x)-s_{data}(x)\}^2]&=E_{data}[s_{data}(x)]^2+E_{data}[s_\theta(x)]^2\\&-2\int s_{data}(x)s_\theta(x)dx\\
&=const.+E_{data}[s_\theta(x)]^2-2p_{data}(x)s_\theta(x)|^\infty_{-\infty}\\
&+2\int p_{data}(x)\nabla _xs_\theta(x)dx\\
assuming:\ x\rightarrow\pm\infty\ \Rightarrow\ &p_{data}(x)\rightarrow0:\\
&\Rightarrow E_{data}[x]^2+2\int p_{data}(x)\nabla_xs_\theta(x)dx+c\\
if\ x\ is\ high-dim\ data:\\
&\Rightarrow E_{data}[||s_\theta(x)||^2_2]+2E_{data}[tr(\nabla_xs_\theta(x))]+c\\
&= E_{data}[||s_\theta(x)||^2_2+2tr(\nabla_xs_\theta(x))]+c\\
&\Rightarrow E_{data}[||s_\theta(x)||^2_2+2tr(\nabla_xs_\theta(x))]
\end{align*}
$$
其中，$\nabla_xs_\theta(x)$是Hessian矩阵，因此求解这个项的开销在$x$的维度较大时会很高。

Score Matching本身基于连续可微等假设，而实际数据往往是离散的，因此引出后续解决这些问题的优化。

#### Denoising Score Matching

对于图像数据，像素值$x_i\in\{0,1,...,255\}$，是离散的，因此我们会添加噪声$\varepsilon\sim\mathcal{N}(0,\sigma^2\rm I)$，添加噪声后的连续光滑结果为$\tilde x=x+\varepsilon$：
$$
q_{data}(\tilde x)=\int q(\tilde x|x)p_{data}(x)dx,\ where\ q(\tilde x|x)\sim\mathcal N(x,\sigma^2\rm I)
$$
由此，将新数据代入$\mathcal L$，得到新Loss：
$$
\begin{align*}
\mathcal L_{Fisher}(q_{data}(\tilde x)||p_\theta(\tilde x))&=E_{q_{data}(\tilde x)}[||\nabla_{\tilde x}{\rm log}\ q_{data}(\tilde x)-\nabla_{\tilde x}{\rm log}\ p_\theta(\tilde x)||^2]\\
&=const.+E_{q(\tilde x|x)}[||\nabla_{\tilde x}{\rm log}\ p_\theta(x)-\nabla_{\tilde x}{\rm log}\ q(\tilde x|x)||^2]\\
&\Rightarrow E_{q(\tilde x|x)}[||\nabla_{\tilde x}{\rm log}\ p_\theta(x)-\nabla_{\tilde x}{\rm log}\ q(\tilde x|x)||^2]

\end{align*}
$$

> 与之同类的方法很多，就没有继续看了。

#### Langevin MCMC

当我们训练出一个$p_\theta(x)$时，我们可以使用Langevin MCMC实现采样。输入一个根据先验分布构造的初始集$x^0$，通过迭代，根据分布梯度$\nabla_x {\rm log}\ p_\theta(x)$，令集合在第$K$步收敛到一个结果：
$$
x^{k+1}\leftarrow x^k+\frac{\epsilon^2}2 \nabla_x {\rm log}\ p_\theta(x^k)+\epsilon z^k,\ k=0,1,...K-1
$$

> 关于$z$，文中没有明说，但根据Langvin dynamic和退火思路，我认为是一个随机扰动集。