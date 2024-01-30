# UltraSound Rendering

## Image Plane Sweep Volume Illumination

Performance:

1. GPU-based
2. Save memory: IPSVI does not require any preprocessing nor does it need to store intermediate results within an illumination volume .
3. High quality: the integration into a GPU-based ray-caster allows for high image quality as well as improved rendering performance by exploiting early ray 

### Introduction 

Past approaches can be roughly divided into two groups:

1. Volume preprocessing
2. Slice-based volume rendering paradigm 

this work proposed ray-casting for rendering.

> similar to step sampling for calculating light ray and shadow ray.
>
> allowing adaptive sampling ,early ray termination and empty space skipping.

### method 

about synchronization problem , because the cuda performance is determined by the slowest kernel. it is significant to use synchronization techniques .

classify the scattering term into two group:

1. single scattering events: simple path integral
2. multiple scattering events: diffusion-like process $\rightarrow $ assume it is a forward scattering phase function, which mean we only sampling the ray over a cone-shaped region (use importance sampling)

## Interactive Dynamic Volume Illumination with Refraction and Caustics

simi-lagrangian backward integration scheme 

## Efficient Stochastic Rendering of Static and Animated Volumes Using Visibility Sweeps