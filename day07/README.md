# Day 6: Train Variational AutoEncoder on CIFAR-10

## What to reproduce:

* `torch4is/my_loss/gaussian_kldiv.py`
* `examples/vae/config.json`
* `examples/vae/model.py`
* `examples/vae/train.py`

## Variational AutoEncoder

1. VAE is totally different to AE: VAE is generative model.
2. AE targets compressing the feature (encoder), while VAE targets generation (decoder).
3. We use (1) reconstruction loss (2) KL-divergence loss for training.
4. We sample from random noise and create samples.
5. Large KL-loss means that generated samples (z ~ N(0, 1)) will be much 'unlikely'.
6. **HW** read VAE paper(https://arxiv.org/pdf/1906.02691.pdf) and understand **ELBO**

## Re-parameterization Trick

1. Random sampling is NOT differentiable.
2. However, we can detour the problem.
   * y ~ N(mu, var)  : not differentiable to mu and var
   * y = mu + var * noise, noise ~ N(0, 1) : differentiable!
   
## Latent Interpolation

1. If latent space (Z) is well-trained, the network should generate similar images for similar z.
2. We can linearly interpolate through Z and see the corresponding images.

## Beta-VAE

1. To create 'factorized' latent representations (https://openreview.net/pdf?id=Sy2fzU9gl)
   * Why do we need 'factorized' version?
2. Loss = Recon-loss + BETA * KL-loss (Original VAE if BETA = 1)
3. Larger BETA encourages disentangled representation.
4. **HW** Try different BETA (0.1 ~ 10) and check the results.