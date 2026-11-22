# Generative Models on Fashion-MNIST

A comparison of six PyTorch generative models trained on Fashion-MNIST: Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, and DDPM. A separate MLP classifier experiment examines why classification features and transposed weights are not sufficient for image generation.

Course: Deep Learning (D6421), Session 07.

## Experiments

| Notebook | Experiment |
|---|---|
| `01_Vanilla_AE.ipynb` | Autoencoder reconstruction |
| `02_Sparse_AE.ipynb` | Reconstruction with an L1 sparsity penalty |
| `03_VAE.ipynb` | Variational autoencoder reconstruction and sampling |
| `04_Vanilla_GAN.ipynb` | Fully connected GAN |
| `05_DCGAN.ipynb` | Convolutional GAN |
| `06_DDPM.ipynb` | Diffusion model with 1,000 denoising steps |
| `07_MLP_Classifier.ipynb` | Classification, weight transposition, and frozen decoder experiments |
| `08_Comparison.ipynb` | Checkpoint loading, figures, FID, and latency benchmarks |

The notebooks are independent for training. The Comparison notebook loads saved weights instead of retraining the models.

## Setup

- Dataset: Fashion-MNIST, 70,000 grayscale images (28 x 28)
- Split: 56,000 training / 7,000 validation / 7,000 test
- Random seed: 42
- Training batch size: 128
- Training epochs: 20
- AE, Sparse AE, and VAE latent dimension: 64
- GAN and DCGAN noise dimension: 100
- DDPM: 1,000 timesteps, linear beta schedule from 0.0001 to 0.02
- Reference environment: Google Colab with NVIDIA Tesla T4

Each notebook recreates the same split using a seeded PyTorch generator.

## Repository layout

```text
Model-Compare/
├── README.md
├── requirements.txt
├── models/
│   ├── 01_Vanilla_AE.ipynb
│   ├── 02_Sparse_AE.ipynb
│   ├── 03_VAE.ipynb
│   ├── 04_Vanilla_GAN.ipynb
│   ├── 05_DCGAN.ipynb
│   ├── 06_DDPM.ipynb
│   ├── 07_MLP_Classifier.ipynb
│   └── 08_Comparison.ipynb
├── outputs/
│   ├── vanilla_ae/
│   ├── sparse_ae/
│   ├── vae/
│   ├── vanilla_gan/
│   ├── dcgan/
│   ├── ddpm/
│   ├── mlp_classifier/
│   └── comparison/
└── checkpoints/     # Optional trained weights; may be hosted separately
```

The notebooks were originally stored in `Model_Comparison/models/` on Google Drive. The `notebooks/` directory above is the proposed GitHub layout. Python modules under `models/` will be added separately to satisfy the assignment's modular-script requirement.

## Running the notebooks

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open a notebook in Google Colab and select a GPU runtime. In Cell 2, use:

```python
USE_GOOGLE_DRIVE = True
```

With this setting, checkpoints and outputs are saved to `/content/drive/MyDrive/Model_Comparison/`. Set it to `False` to use the local project directory. Changing the setting does not copy existing checkpoints between locations.

To reproduce training, run the model notebooks in order. This will train models again and may overwrite previously saved checkpoint files.

To evaluate the already trained models, make sure the checkpoint files are available in `checkpoints/`, then run `08_Comparison.ipynb`. It loads the saved weights and does not retrain any models. DDPM inference and FID evaluation are still computationally expensive.

## Results

Results below are from the completed Colab run. Reconstruction metrics were evaluated on the held-out test split. FID uses 1,000 real test images and 1,000 generated images per generative model.

| Model | Test MSE | PSNR (dB) | SSIM | FID-2048 |
|---|---:|---:|---:|---:|
| Vanilla AE | 0.007078 | 22.5646 | 0.8158 | N/A |
| Sparse AE | 0.009742 | 21.0562 | 0.7597 | N/A |
| VAE | 0.017936 | 18.1290 | 0.6315 | 107.3204 |
| Vanilla GAN | N/A | N/A | N/A | 197.2945 |
| DCGAN | N/A | N/A | N/A | 47.7320 |
| DDPM | N/A | N/A | N/A | 48.0745 |

The MLP classifier reached **89.33%** test accuracy. Its frozen decoder obtained a test reconstruction MSE of **0.018117**, compared with **0.007078** for Vanilla AE.

### Inference latency

Median latency for a batch of 64 images on Tesla T4, measured across three runs after warm-up:

| Model | Latency (ms) | Mean training time (s/epoch) |
|---|---:|---:|
| Vanilla AE | 0.454 | 7.524 |
| Sparse AE | 0.807 | 7.745 |
| VAE | 0.375 | 7.626 |
| Vanilla GAN | 1.053 | 8.037 |
| DCGAN | 3.299 | 14.367 |
| DDPM | 15034.586 | 23.888 |

AE and Sparse AE measure reconstruction, while VAE, GAN, DCGAN, and DDPM measure image synthesis. DDPM samples 64 images in chunks of 16, using 1,000 reverse diffusion steps per chunk.

FID was calculated from 2,048-dimensional InceptionV3 features, with grayscale images converted to three channels. The small sample size and the difference between Fashion-MNIST and InceptionV3's training domain limit how broadly these values can be interpreted.

## Outputs

`outputs/comparison/` contains the consolidated results:

- `final_metrics_summary.csv`: reconstruction scores, FID, latency, and parameter counts
- `fid_scores.csv`: FID for VAE, Vanilla GAN, DCGAN, and DDPM
- `benchmark_results_final.csv`: inference latency and training time
- `benchmark_raw_measurements.csv`: individual latency measurements
- `reconstruction_metrics.csv`: AE, Sparse AE, and VAE reconstruction scores
- `figure_1_reconstruction.png`: reconstruction comparison
- `figure_2_random_sampling.png`: generated samples
- `figure_3_mlp_limitations.png`: classifier inversion and decoder experiments
- `figure_4_interpolation.png`: latent interpolation
- `figure_5_training_losses.png`: training curves
- `figure_5_ddpm_trajectory.png`: reverse diffusion trajectory

The individual model output folders contain their own training histories, metrics, and figures.

## Notes

Checkpoints are necessary for inference without retraining. They are not embedded in the notebooks. Fixed seeds reduce variability but do not guarantee identical results across PyTorch versions and GPU environments.

The written report is a separate deliverable and covers architecture diagrams, loss functions, empirical results, and the assignment questions Q1 to Q6.
