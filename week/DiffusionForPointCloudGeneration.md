> 本周工作：
>
> 1. 轮胎标记完成数据集1的标记，开展对数据集2的标记；
> 2. 研读需审核的论文paper_563，根据它的引用完成了对它实现思路的理解与整理，并和学长对此进行了讨论；
> 3. unity调研，对屠老师发的ppt涉及的技术进行了浏览与了解；
> 4. 粗略了解了Mesh Denoise的流程；
> 5. 完成了对diffusion Model的Loss的推导理解，对Diffusion Probabilistic Models for 3D Point Cloud Generation进行了学习

# Diffusion Probabilistic Models for 3D Point Cloud Generation

## Introduction

该文受非平衡热力学启发，基于DiffusionModel提出了点云生成方法。其中，他们认为3D点云中的点可视为非平衡热力学系统中的粒子，在扩散作用下粒子会从某形状扩散到整个空间。这个工作将点云的点分布和噪声分布建立关联。通过学习寻找逆分布，从而从噪声中恢复原始点分布。

<img src="C:\Users\12313123\AppData\Roaming\Typora\typora-user-images\image-20221105151234690.png" alt="image-20221105151234690" style="zoom:50%;" />

## Diffusion Probabilistic Models for Point Clouds

这部分定义了模型训练的前向和后向扩散的概率模型，最后定义了训练Loss。

### Formulation

遵循热力学和点云定义，定义点云$ X^{(0)}=\{x_i^{(0)}\}^N_{i=1}$为一组热力学系统中的粒子，每一个粒子$x_i$都可被独立采样于点分布$q(x_i^{(0)}|z)$，其中$z$为Shape Latent（决定点分布）。

遵循Diffusion，点到随机噪声这一扩散过程可描述为以下蒙特卡洛链：
$$
\begin{align*}
q(x^{(1:T)}_i|x_i^{(0)})&=\prod^T_{t=1}q(x_i^{(t)}|x_i^{(t-1)})\\
where\ q(x^{(t)}|x^{(t-1)})&=\mathcal N(x^{(t)}|\sqrt{1-\beta_t}x^{(t-1)},\beta_t{\bf I}),\ t=1,...,T
\end{align*}
$$
由于目标是根据潜在编码$z$生成点云，因此定义逆向扩散过程。

1. 从近似于$q(x_i^{(t)})$的分布$q(x_i^{(t)})$采样一组点作为输入；
2. 通过蒙特卡洛链逆向回期望模型；

相比于前向的简单加噪声，逆向的模型是未知的，需要学习的。逆向过程可描述为：
$$
\begin{align*}
p_\theta(x^{(0:T)}|z)&=p(x^{(T)})\prod^T_{i=1}p_\theta(x^{(t-1)}|x^{(t)},z)\\
p_\theta(x^{(t-1)}|x^{(t)},z)&=\mathcal N(x^{(t-1)}|\mu_\theta(x^{(t)},t,z),\beta_t{\bf I}),\ 
where\ p(x^{(T)})\sim \mathcal N(0,{\bf I})
\end{align*}
$$
由于输入点云是从分布$p(x_i^{t})$中采样的，因此整个点云的概率就是所有样本点的乘积：
$$
\begin{align*}
q(X^{(1:T)}|X^0)=\prod^N_{i=1}q(x_i^{(1:T)}|x_i^{(0)})\\
p_\theta(X^{(1:T)}|z)=\prod^N_{i=1}p_\theta(x_i^{(1:T)}|z)
\end{align*}
$$

### Training Objective

训练反向扩散的目的是使点云的似然估计$\mathbb E[{\rm log}\ p_\theta(X^{(0)})]$最大化。但直接算不行（见EBM），因此使用最大下界化简：
$$
\begin{align*}

\mathbb E[{\rm log}\ p_\theta(X^{(0)})]&\ge \mathbb E_q\Bigg[{\rm log}\ \frac{p_\theta(X^{(0:T)},z)}{q(X^{(1:T)},z|X^{(0)})}\Bigg]\\
&=E_q\Bigg[{\rm log}\ p(X^{(T)})\\
&+\sum^T_{t=1}{\rm log}\ \frac{p_\theta(X^{(t-1)}|X^{(t)},z)}{q(X^{(T)}|X^{(t-1)})}\\
& -{\rm log}\ \frac{q_\varphi(z|X^{(0)})}{p(z)}\Bigg]

\end{align*}
$$

> TODO：继续完成研读。

### Model Implementations

<img src="C:\Users\12313123\AppData\Roaming\Typora\typora-user-images\image-20221105171115785.png" alt="image-20221105171115785" style="zoom:67%;" />