# Generative Model Comparison on Fashion-MNIST

PyTorch experiments comparing six generative models on Fashion-MNIST: Vanilla Autoencoder (AE), Sparse Autoencoder, Variational Autoencoder (VAE), Vanilla GAN, DCGAN, and DDPM.

A separate MLP classifier experiment tests whether classifier weights and latent features can be used for image reconstruction or generation. The results are collected in a comparison notebook.

Course: Deep Learning (D6421), Session 07.

## Repository structure

```text
Model-Compare/
├── README.md
├── requirements.txt
├── models/
│   ├── Vanilla_AE.ipynb
│   ├── Sparse_AutoEncoder.ipynb
│   ├── Variational AutoEncoder.ipynb
│   ├── VanillaGAN.ipynb
│   ├── DCGAN.ipynb
│   ├── DDPM.ipynb
│   ├── MLP_Classifier.ipynb
│   └── Comparison.ipynb
├── checkpoints/
│   ├── vanilla_ae.pt
│   ├── sparse_ae.pt
│   ├── vae.pt
│   ├── vanilla_gan_generator.pt
│   ├── vanilla_gan_discriminator.pt
│   ├── dcgan_generator.pt
│   ├── dcgan_discriminator.pt
│   ├── ddpm.pt
│   ├── mlp_classifier.pt
│   ├── mlp_frozen_decoder.pt
│   └── mlp_latent_prior.pt
└── outputs/
    ├── vanilla_ae/
    ├── sparse_ae/
    ├── vae/
    ├── vanilla_gan/
    ├── dcgan/
    ├── ddpm/
    ├── mlp_classifier/
    └── comparison/
```

The notebooks contain the model definitions, training or evaluation code, and saved execution outputs. The `models/` directory is named as it is in the repository; it contains `.ipynb` files rather than importable Python modules.

## Experimental setup

| Setting | Value |
|---|---|
| Dataset | Fashion-MNIST, 70,000 grayscale images (28 x 28) |
| Split | Train 56,000 / validation 7,000 / test 7,000 |
| Random seed | 42 |
| Training batch size | 128 |
| Training epochs | 20 per generative model |
| AE, Sparse AE, VAE latent size | 64 |
| Vanilla GAN and DCGAN input noise | 100 |
| DDPM timesteps | 1,000 |
| Reference environment | Google Colab, NVIDIA Tesla T4 |

The split is created by combining the original Fashion-MNIST train and test datasets and applying `torch.utils.data.random_split` with seed 42. All experiments use the same partition indices.

## Running the experiments

The notebooks were developed in Google Colab. A CUDA-enabled runtime is recommended, particularly for DDPM sampling and FID evaluation.

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Each notebook has a storage setting in its second code cell:

```python
USE_GOOGLE_DRIVE = True
```

When enabled, the notebooks read and write files under:

```text
/content/drive/MyDrive/Model_Comparison/
├── checkpoints/
└── outputs/
```

Before running the comparison notebook against existing results, copy the repository's `checkpoints/` and `outputs/` folders into that Google Drive directory. The directory names and checkpoint filenames must remain unchanged. A local alternative is `USE_GOOGLE_DRIVE = False`, which uses the notebook runtime's current working directory as the project root. Some training notebooks also import `google.colab`, so they are intended primarily for Colab.

### Training from scratch

Open the individual notebooks in `models/` and run their cells in order. Each experiment trains and saves its own model weights and results. There is no required order among the six generative-model training notebooks, but the comparison requires their saved checkpoints. `MLP_Classifier.ipynb` supplies the additional classifier checkpoints and results.

**Running the training notebooks again will retrain the models and may overwrite existing checkpoint files.** Preserve a backup of the completed run if you want to compare results.

### Evaluating saved checkpoints

Open `models/Comparison.ipynb` in Colab and run it in cell order after making the 11 checkpoint files and the model output histories available at the configured project path.

The comparison notebook loads pretrained model weights; it does not retrain the six generative models. However, generating DDPM samples and recomputing FID still require significant GPU time. Existing evaluation results and figures can be inspected directly in `outputs/comparison/` without executing the notebook.

## Results

The following values come from the completed Tesla T4 run, as stored in `outputs/comparison/final_metrics_summary.csv`.

Reconstruction metrics were evaluated on the held-out test partition. FID-2048 was calculated using 1,000 real and 1,000 generated images per model for VAE, Vanilla GAN, DCGAN, and DDPM.

| Model | Test MSE | PSNR (dB) | SSIM | FID-2048 |
|---|---:|---:|---:|---:|
| Vanilla AE | 0.007078 | 22.5646 | 0.8158 | N/A |
| Sparse AE | 0.009742 | 21.0562 | 0.7597 | N/A |
| VAE | 0.017936 | 18.1290 | 0.6315 | 107.3204 |
| Vanilla GAN | N/A | N/A | N/A | 197.2945 |
| DCGAN | N/A | N/A | N/A | 47.7320 |
| DDPM | N/A | N/A | N/A | 48.0745 |

The MLP classifier achieved 89.33% test accuracy. Its decoder trained on frozen classifier features obtained a reconstruction MSE of approximately 0.018117, compared with 0.007078 for Vanilla AE.

### Runtime

The table reports the recorded mean training time per epoch and median batch-64 inference latency. Measurements were taken on a Tesla T4.

| Model | Training time (s/epoch) | Latency (ms/batch of 64) |
|---|---:|---:|
| Vanilla AE | 7.52 | 0.454 |
| Sparse AE | 7.75 | 0.807 |
| VAE | 7.63 | 0.375 |
| Vanilla GAN | 8.04 | 1.053 |
| DCGAN | 14.37 | 3.299 |
| DDPM | 23.89 | 15034.586 |

AE and Sparse AE latency measures reconstruction rather than sampling from an unconditional prior. DDPM latency includes iterative sampling over 1,000 reverse steps and was measured in smaller chunks to produce a total batch of 64 images.

FID uses InceptionV3 features on grayscale images converted to three channels. With only 1,000 generated samples per model, close FID scores should not be interpreted as a definitive ranking.

## Evaluation artifacts

The `outputs/comparison/` folder contains:

| File | Description |
|---|---|
| `final_metrics_summary.csv` | Combined comparison metrics and parameter counts |
| `reconstruction_metrics.csv` | MSE, PSNR, and SSIM for AE, Sparse AE, and VAE |
| `fid_scores.csv` | FID scores and sample counts |
| `benchmark_results_final.csv` | Training-time and batch-64 inference benchmark |
| `benchmark_raw_measurements.csv` | Individual benchmark measurements |
| `figure_1_reconstruction.png` | Original images and model reconstructions |
| `figure_2_random_sampling.png` | Randomly generated samples |
| `figure_3_mlp_limitations.png` | MLP weight-transpose, decoder, and sampling comparison |
| `figure_4_interpolation.png` | Latent interpolation |
| `figure_5_training_losses.png` | Training loss curves |
| `figure_5_ddpm_trajectory.png` | DDPM reverse denoising trajectory |

Each model's output directory also contains experiment-specific CSV files and figures.

## Reproducibility

The experiments use fixed random seeds and consistent dataset splits. Results may still vary between runs because of hardware, CUDA, and library behavior.

The committed checkpoints allow the trained models to be loaded without running training again. The included notebook outputs and CSV files document the completed run; they are not a substitute for running the code if independent reproduction is required.
