# Comparative Study of Generative Models

Empirical comparison of six generative deep learning architectures on **Fashion-MNIST**: Vanilla Autoencoder (AE), Sparse Autoencoder, Variational Autoencoder (VAE), Vanilla GAN, Deep Convolutional GAN (DCGAN), and Denoising Diffusion Probabilistic Model (DDPM).

The project also investigates why the weights and latent features of a supervised **MLP classifier** cannot directly serve as a reliable image generator. Experiments include weight-transpose projection, reconstruction through a decoder trained on frozen classifier features, and Gaussian sampling from the classifier's empirical latent distribution.

This repository accompanies a take-home assignment for **Deep Learning (D6421), Session 07**.

## Dataset and experimental setup

| Setting | Value |
|---|---|
| Dataset | Fashion-MNIST, grayscale 28 × 28, 10 classes |
| Total images | 70,000 |
| Train / validation / test | 56,000 / 7,000 / 7,000 (80% / 10% / 10%) |
| Random seed | 42 |
| Training batch size | 128 |
| Epochs | 20 per generative model |
| Shared AE/VAE latent dimension | 64 |
| Vanilla GAN / DCGAN noise dimension | 100 |
| DDPM noise shape | 1 × 28 × 28 |
| Reference execution hardware | NVIDIA Tesla T4 GPU (Google Colab) |

The notebook creates one deterministic dataset split shared by all experiments. Training uses `shuffle=True`; validation and test loaders use `shuffle=False`.

## Models and evaluations

- **Vanilla AE:** image reconstruction using a deterministic bottleneck.
- **Sparse AE:** image reconstruction with a sparsity penalty on latent representations.
- **VAE:** probabilistic latent encoding, reconstruction, and prior sampling.
- **Vanilla GAN:** fully connected adversarial generator and discriminator.
- **DCGAN:** convolutional adversarial generator and discriminator.
- **DDPM:** iterative denoising from Gaussian noise.
- **MLP classifier:** a separate 784 → 256 → 64 → 10 classifier, evaluated for classification accuracy and limitations in image reconstruction and generation.

Reconstruction is evaluated with **test MSE, PSNR, and SSIM** (AE, Sparse AE, VAE). Generative quality is compared using **FID-2048** with 1,000 real test images and 1,000 generated images per model (VAE, Vanilla GAN, DCGAN, DDPM). The runtime benchmark measures inference latency for a **batch of 64 images**, including full iterative sampling for DDPM.

## Getting started in Google Colab

1. Open the executed notebook, **`Code (1).ipynb`**, in Google Colab. You may rename it to `Code.ipynb` when uploading to GitHub; update this README if you do.
2. Select **Runtime → Change runtime type → T4 GPU** (or another CUDA-enabled GPU, if available).
3. Confirm the first cell installs the required FID packages **before** importing them. The notebook includes `%pip install -q "torchmetrics[image]" torch-fidelity`.
4. Select **Runtime → Restart session and run all**. This performs dataset download, training, evaluation, FID, and figure generation from scratch. **It is computationally expensive**, especially DDPM training, sampling, and FID.
5. Download the generated `outputs/` folder after completion. Files stored in the Colab runtime are not permanent.

For a local Python environment, install the dependencies first:

```bash
python -m pip install -r requirements.txt
```

Then launch Jupyter and run the notebook with a compatible PyTorch/CUDA environment. The executed notebook was validated on a Colab Tesla T4; results and runtimes may vary on other hardware and software versions.

## Outputs

The notebook writes its results into `outputs/`:

| File | Content |
|---|---|
| `figure_1_reconstruction.png` | Original test images versus AE, Sparse AE, and VAE reconstructions |
| `figure_2_random_sampling.png` | 4 × 4 random samples from each of VAE, GAN, DCGAN, and DDPM |
| `figure_3_mlp_limitations.png` | Original images, weight-transpose projection, frozen decoder comparison, and latent-prior samples |
| `figure_4_interpolation.png` | Latent interpolation for AE, Sparse AE, VAE, GAN, and DCGAN |
| `figure_5_training_losses.png` | Training loss curves for all six generative architectures |
| `figure_5_ddpm_trajectory.png` | DDPM reverse-denoising trajectory |
| `report_metrics.csv` | Combined reconstruction, FID, training-time, and latency metrics |
| `fid_scores.csv` | Per-model FID scores and sample counts |
| `benchmark_results_final.csv` | Per-model training times and batch-64 inference benchmarks |
| `benchmark_raw_measurements.csv` | Raw timing measurements |
| `mlp_weight_invertibility.csv` | Numerical comparisons of classifier weight Gram matrices with identity |
| `model_parameters.csv` | Model component parameter counts |

The outputs for the completed reference run are also available in `outputs.zip` if included with the repository.

## Reference results

These figures come from the completed **Google Colab Tesla T4** run, rather than from a claim that these architectures always rank the same way.

| Model | Test MSE | Test PSNR (dB) | Test SSIM | FID-2048 (1,000 samples) |
|---|---:|---:|---:|---:|
| Vanilla AE | 0.007096 | 22.5605 | 0.8150 | N/A |
| Sparse AE | 0.009968 | 20.9573 | 0.7558 | N/A |
| VAE | 0.017914 | 18.1377 | 0.6324 | 108.6526 |
| Vanilla GAN | N/A | N/A | N/A | 183.9383 |
| DCGAN | N/A | N/A | N/A | 47.3670 |
| DDPM | N/A | N/A | N/A | 71.3775 |

A lower FID indicates greater similarity according to Inception features. **FID was computed using only 1,000 synthetic samples per model on 28 × 28 grayscale Fashion-MNIST images converted to three channels.** It should not be compared directly with large-scale natural-image FID benchmarks. The MLP classifier obtained **89.33% test accuracy** in the reference run.

Additional latency and training-time measurements are in `outputs/report_metrics.csv`. GPU timings do not establish performance on mobile devices.

## Reproducibility notes

- Set all random seeds as defined in the notebook before training.
- Data partitions are fixed with a seeded PyTorch generator.
- The full notebook is designed to be executed in order from the first cell.
- Runtime, CUDA, library versions, and GPU nondeterminism may produce small differences in measured times or generated samples.
- `outputs/` contains evaluation artifacts; it does not contain trained model weights.

## Repository organization

```text
Model-Compare/
├── Model_Comparison.ipynb
├── README.md
├── requirements.txt
├── outputs/                     # Generated figures and CSV results (optional)
│   ├── figure_1_reconstruction.png
│   ├── figure_2_random_sampling.png
│   ├── figure_3_mlp_limitations.png
│   ├── figure_4_interpolation.png
│   ├── figure_5_training_losses.png
│   ├── figure_5_ddpm_trajectory.png
│   └── ...
└── outputs.zip                  # Optional archive of generated outputs
```
