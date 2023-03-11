# 3-12周报

> 本周工作：
>
> 1. Foundation Model 调研任务；
> 2. 点云（和 Mesh）数据集收集；
> 3. 寻找梯度估计的灵感：收集传统点云降噪论文和与3D梯度有关的深度论文；
> 4. 完善公式推导，找到了一个可尝试的修正了参数的新loss。
> 5. 对模型部分代码优化，在不影响效率下大大降低了GPU显存占用；
>
> 周报内容是对 工作3 和 工作4 的阐述。
>
> 关于上周提出的 **对于 $x^{(t)}_a$ 生成策略的修改**，训练实验结果显示不合理，且基于 $Nearest$ 生成的结果显然不通过 Jarque-Bera 检验。（注：原方法的中间变量均通过JB检验）

## 估计梯度的已有思路





## 公式推导与分析

> 对部分符号进行重定义，为了降低公式编写难度，例如 $x^{(t)}$ 重定义为了 $x^t$。

### SBD Loss推导

定义 Diffusion Process 的分布描述：
$$
q(x^{1:T}|x^0)=\prod^T_{t=1}q(x^t|x^{t-1}),\ q(x^t|x^{t-1})=\mathcal N(x^t;\ \sqrt{1-\beta_t}x^{t-1},\ \beta_t{\rm I})
$$
定义 Sampling Process 的分布描述：（带 $\theta$ 的为需要训练的
$$
\begin{align*}
p_\theta(x^{0:T}|F_{T})&=p(x^T)\prod^T_{t=1}p_\theta(x^{t-1}|x^{t},F_{T})\\
where\ p_\theta(x^{t-1}|x^{t},F_{T})&=\mathcal N(x^t;\ \sqrt{1-\beta_t}x^{t-1},\ \beta_t{\rm I}),\ p(x^T)=\mathcal N(x^T;x^0,L_{noise}{\rm I^3})
\end{align*}
$$
其中，$F_T=EdgeFeature(x^T)$

训练方法 $p_\theta$ 使用它的负对数似然估计的变分上界：（**绿色部分**为相比于上行公式**修改的部分**，目的是降低阅读难度）
$$
\begin{align*}
\mathbb E[-\log p_\theta(x^0)]&\le \mathbb E_q\bigg[-\log \frac {p_\theta(x^{0:T}|F_{T})}{q(x^{1:T}|x^0)}\bigg]\\
&= E_q\bigg[-\log p(x^T)-\sum_{\color{green}t\ge1}\log \frac {p_\theta(x^{t-1}|x^{t},F_{T})}{q(x^t|x^{t-1})}\bigg]\\
&=E_q\bigg[-\log p(x^T)-\sum_{\color{green}t> 1}\log \frac {p_\theta(x^{t-1}|x^{t},F_{T})}{q(x^t|x^{t-1})}-{\color{green}\log\frac {p_\theta(x^{0}|x^{1},F_{T})}{q(x^1|x^{0})}}\bigg]\\
\end{align*}
$$
根据贝叶斯定理: $q(x^t|x^{t-1})=\cfrac{q(x^{t-1}|x^t)q(x^t)}{q(x^{t-1})}$，但是 $q(x^{t-1}|x^t)$ 并不合理。又因 $q(x^{1:T}|x^0)$ 令 $q(x^t|x^{t-1},x^0)=q(x^t|x^{t-1})$ 满足，因此引入 $x^0$ 作为条件 $q(x^t|x^{t-1})=q(x^t|x^{t-1},x^0)=\cfrac{q(x^{t-1}|x^t,x^0)q(x^t|x^0)}{q(x^{t-1}|x^0)}$。
$$
\begin{align*}
&=E_q\bigg[-\log p(x^T)-\sum_{t> 1}\log \frac {p_\theta(x^{t-1}|x^{t},F_{T})}{\color{green}q(x^{t-1}|x^t,x^0)}\cdot \frac{\color{green}q(x^{t-1}|x^0)}{\color{green}q(x^{t}|x^0)}-{\log\frac {p_\theta(x^{0}|x^{1},F_{T})}{q(x^1|x^{0})}}\bigg]\\
\because \ &\sum_{t>1}\log \bigg[\frac{q(x^{t-1}|x^0)}{q(x^{t}|x^0)}\bigg]=\log q(x^1|x^0)-\log q(x^T|x^0)\\
\therefore\ &=E_q\bigg[-\log \frac{p(x^T)}{\color{green}q(x^T|x^0)}-\sum_{t> 1}\log \frac {p_\theta(x^{t-1}|x^{t},F_{T})}{q(x^{t-1}|x^t,x^0)}-{\log {\color{green}p_\theta(x^{0}|x^{1},F_{T})}}\bigg]\\
&= E_q\bigg[D_{KL}({q(x^T|x^0)}\ ||\ {p(x^T)})+\sum_{t> 1}D_{KL}({q(x^{t-1}|x^t,x^0)} \ ||\ {p_\theta(x^{t-1}|x^{t},F_{T})})-{\log {p_\theta(x^{0}|x^{1},F_{T})}}\bigg]\\
&={\color{red}D_{KL}({q(x^T|x^0)}\ ||\ {p(x^T)})}+E_q\bigg[\sum_{t> 1}D_{KL}({q(x^{t-1}|x^t,x^0)} \ ||\ {p_\theta(x^{t-1}|x^{t},F_{T})})\bigg]-{\color{red}\log {p_\theta(x^{0}|x^{1},F_{T})}}\\
&=:loss
\end{align*}
$$
其中，红色项显然是常数项，对Loss的下降并不会起到任何作用，因此带 $F_T$ 的 Diffusion 最优化问题可描述为：
$$
\begin{align*}
Simplify&\Rightarrow loss=E_q\bigg[\sum_{t> 1}D_{KL}({q(x^{t-1}|x^t,x^0)} \ ||\ {p_\theta(x^{t-1}|x^{t},F_{T})})\bigg]\\
&\Leftrightarrow \mathop{\arg\min}\limits_{\theta}\ E_q\bigg[\sum_{t> 1}D_{KL}({q(x^{t-1}|x^t,x^0)} \ ||\ {p_\theta(x^{t-1}|x^{t},F_{T})})\bigg]
\end{align*}
$$
结论：

- 引入 $F_T$ 只影响 ${q(x^{t-1}|x^t,x^0)}$ 和 ${p_\theta(x^{t-1}|x^{t},F_{T})}$ 的相对熵；

#### 引入 Score-based 

这里开始，我引入 Score-based 作为 $p_\theta$ 中计算梯度的模型，把上面的最优化问题进行限制。定义 Score-based 计算梯度：
$$
\sqrt{\bar\alpha_t}\nabla_x log[s_\theta(x^{t-1}_a|x^{t}_a,F_T)]\approx -z_\theta\propto\min\{||x^{0}_i-x^{t}_{a}||^2_2\ |x^{0}_i\in x^{0}\},\ x^{t}_{a}=\frac{x^{t}}{\sqrt{\overline\alpha_t}}
$$ {t}
Sampling Process可描述为：
$$
x^{t-1}=\frac 1 {\sqrt {\alpha_t}}(x^t-\frac{\beta_t\sqrt{\bar\alpha_t}}{\sqrt{1-\overline a_t}}(-\nabla_x log[s_\theta(x^{t-1}_a|x^{t}_a,F_T)]))\\
\Rightarrow x^t=\sqrt{\alpha_t}x^{t-1}+\frac{\beta_t\sqrt{\bar\alpha_t}}{\sqrt{1-\overline a_t}}(-\nabla_x log[s_\theta(x^{t-1}_a|x^{t}_a,F_T)])
$$
又因为，对于 Diffusion process 来说：
$$
x^{t}=\sqrt{\alpha_t}x^{t-1}+\sqrt{1-\alpha_t}z,\ z\sim\mathcal N(0,{\bf I})
$$
对于 KL 散度来说，我们可以使用 MSE 计算 KL 散度：
$$
\begin{align*}
D_{KL}({q} \ ||\ {p_\theta})&\propto \frac 1 2 \bigg|\bigg|(\sqrt{\alpha_t}x^{t-1}+\sqrt{1-\alpha_t}z)-\Big(\sqrt{\alpha_t}x^{t-1}+\frac{\beta_t\sqrt{\bar\alpha_t}}{\sqrt{1-\overline a_t}}(-\nabla_x log[s_\theta(x^{t-1}_a|x^{t}_a,F_T)])\Big)\bigg|\bigg|^2_2\\
&=\frac 1 2 \bigg|\bigg|\sqrt{1-\alpha_t}z+\frac{\beta_t\sqrt{\bar\alpha_t}}{\sqrt{1-\overline a_t}}\nabla_x log[s_\theta(x^{t-1}_a|x^{t}_a,F_T)]\bigg|\bigg|^2_2\\
\because p(x;\mu,\Sigma)&=\frac 1{\sqrt{(2\pi)^n|\Sigma|}}e^{-\frac{(x-\mu)^T\Sigma^{-1}(x-\mu)}{2}}
\propto
p(x;\mu,\sigma^2)=\frac 1{\sqrt{(2\pi)^n}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}},when\ \Sigma=\sigma^2{\rm I}\\
\therefore\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ &\Rightarrow \frac 1 2 \bigg|\bigg|\sqrt{1-\alpha_t}(-\nabla_xq(x^t))+\frac{\beta_t\sqrt{\bar\alpha_t}}{\sqrt{1-\overline a_t}}\nabla_x log[s_\theta(x^{t-1}_a|x^{t}_a,F_T)]\bigg|\bigg|^2_2
\end{align*}
$$
综上所述：
$$
\mathcal L(x^{0:T},\{\beta_i\}^T_{i=1})=\frac 1 2\sum_{t>1}\mathbb E_{q}\bigg[\bigg|\bigg|\frac{\beta_t\sqrt{\bar\alpha_t}}{\sqrt{1-\overline a_t}}\nabla_x log[s_\theta(x^{t-1}_a|x^{t}_a,F_T)]-\sqrt{1-\alpha_t}\nabla_xq(x^t)\bigg|\bigg|^2_2\bigg]
$$
结论：

- loss形式上和当前使用的loss一致，但细节参数不同，下周会对这个新loss进行验证。

