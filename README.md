# 3D-fAnoGAN for Prostate Anomaly Detection

## Overview
This project implements a 3D Fast Anomaly GAN (f-AnoGAN) for detecting anomalies in prostate CT scans. The implementation is inspired by the original f-AnoGAN paper [1] and extends it to work with 3D medical imaging data.

⚠️ **Note**: This is a work in progress. Regular updates and improvements will be made to enhance the model's performance and capabilities.

## Current Features
- 3D Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation
- Spectral normalization for training stability
- Adaptive gradient penalty weighting
- Comprehensive anomaly scoring system
- Separate reconstruction and feature-based error metrics
- Multiple thresholding methods for anomaly detection
- Visualization tools for analysis

## Model Architecture
The current implementation includes:
- A Generator network with 3D convolutional layers
- A Discriminator with feature extraction capabilities
- An Encoder network for mapping to latent space
- Advanced stabilization techniques for GAN training

## Performance Features
- Batch processing for memory efficiency
- GPU acceleration support
- Configurable model parameters
- Multiple evaluation metrics

## Future Updates
- [ ] Enhanced architecture with residual connections
- [ ] Self-attention mechanisms
- [ ] Progressive growing capabilities
- [ ] Additional visualization tools
- [ ] Performance optimization
- [ ] Extended documentation

## Reference
[1] Schlegl, T., Seeböck, P., Waldstein, S.M., Langs, G., & Schmidt-Erfurth, U. (2019). f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks. Medical Image Analysis, 54, 30-44.

*Last Updated: January 2025*
