# Model Caching System

## Overview

The model training pipeline now includes an intelligent caching system that saves trained models and intermediate data to disk. When you run the training pipeline again with the same parameters, the system will automatically load the cached models instead of retraining them, significantly reducing training time and computational resources.

## Cache Directory

All cached models and data are stored in the `model_cache/` directory in the project root. This directory is automatically created when you first import the source module.

```
project_root/
├── model_cache/
│   ├── xgb_*.pkl              # Cached XGBoost models
│   ├── cvae_*                 # Cached CVAE full models
│   ├── cvae_encoder_*         # Cached CVAE encoder
│   ├── cvae_decoder_*         # Cached CVAE decoder
│   ├── smote_data_*.pkl       # Cached SMOTE resampled data
│   ├── tstr_synth_*.pkl       # Cached TSTR synthetic data
│   └── tstr_model_*.pkl       # Cached TSTR XGBoost models
```

## Cached Components

### 1. XGBoost Models (`train_and_evaluate_xgboost`)
**Cached Data**: Trained XGBoost classifier model
**Cache Key Based On**:
- Model name (e.g., "Baseline", "SMOTE Augmented", "VAE Augmented")
- Training data shape (number of samples and features)
- Number of positive samples
- Total samples

**Benefits**: Avoids retraining XGBoost models (~30-60 seconds per model)

### 2. CVAE Models (`train_cvae`)
**Cached Data**: 
- Full CVAE model
- Encoder model
- Decoder model

**Cache Key Based On**:
- Input dimension
- Latent dimension
- Number of classes
- Number of epochs
- Batch size
- Training data size

**Benefits**: Avoids retraining the CVAE (~5-20 minutes depending on epochs and data size)

### 3. SMOTE Data (`apply_smote_and_train_xgboost`)
**Cached Data**: SMOTE resampled X and y data
**Cache Key Based On**:
- Training data shape
- Number of positive samples
- Random state

**Benefits**: Avoids SMOTE resampling computation (~10-30 seconds)

### 4. TSTR Protocol (`run_tstr_protocol`)
**Cached Data**: 
- Synthetic data generated for TSTR
- Trained TSTR XGBoost model

**Cache Key Based On**:
- Number of synthetic samples per class
- Latent dimension
- Random state
- Input dimension

**Benefits**: Avoids synthetic data generation and model training (~2-5 minutes)

## How It Works

1. **Cache Key Generation**: Each function generates a unique cache key using MD5 hashing of relevant parameters
2. **Cache Lookup**: Before training, the function checks if a cached version exists
3. **Load or Train**: 
   - If cache exists: Load the cached model/data and skip training
   - If cache doesn't exist: Train the model and save it to cache
4. **Automatic Saving**: Newly trained models are automatically saved to cache for future use

## Usage

No changes to your existing code are required! The caching system works transparently:

```python
from source import train_cvae, train_and_evaluate_xgboost

# First run: trains and caches the model
model, encoder, decoder, latent_dim = train_cvae(X_train, y_train, epochs=100)

# Second run with same parameters: loads from cache instantly
model, encoder, decoder, latent_dim = train_cvae(X_train, y_train, epochs=100)
```

## Cache Management

### Viewing Cache Contents
```bash
ls -lh model_cache/
```

### Clearing Cache
To force retraining of all models, simply delete the cache directory:
```bash
rm -rf model_cache/
```

Or remove specific cached models:
```bash
# Remove only CVAE models
rm model_cache/cvae_*

# Remove only XGBoost models
rm model_cache/xgb_*
```

### Cache Invalidation
The cache is automatically invalidated when:
- Training data size changes
- Model hyperparameters change (epochs, batch_size, latent_dim, etc.)
- Feature dimensions change

## Performance Improvements

Expected time savings on subsequent runs:

| Component | First Run | Cached Run | Time Saved |
|-----------|-----------|------------|------------|
| CVAE Training (100 epochs) | ~10-20 min | ~2-5 sec | ~99% |
| XGBoost Baseline | ~30-60 sec | ~1 sec | ~98% |
| SMOTE Data Generation | ~10-30 sec | ~1 sec | ~97% |
| TSTR Protocol | ~2-5 min | ~2-5 sec | ~98% |
| **Total Pipeline** | ~15-30 min | ~10-20 sec | **~98%** |

## Best Practices

1. **During Development**: Keep cache enabled to iterate quickly on non-model code
2. **Hyperparameter Tuning**: Clear cache when changing model parameters to ensure fresh training
3. **Production**: Consider versioning your cache directory alongside model versions
4. **Disk Space**: Monitor cache size; each CVAE model can be 1-10 MB depending on architecture

## Troubleshooting

### Cache Not Loading
- Verify the cache key matches (same parameters)
- Check file permissions on `model_cache/` directory
- Ensure sufficient disk space

### Want to Force Retrain
```python
# Option 1: Delete specific cache files manually
import os
os.remove('model_cache/cvae_<hash>.h5')

# Option 2: Delete entire cache
import shutil
shutil.rmtree('model_cache/')
```

### Memory Issues
Large cached files can consume disk space. Monitor with:
```bash
du -sh model_cache/
```

## Technical Details

- **XGBoost Models**: Saved using `joblib` for efficient serialization
- **TensorFlow Models**: Saved using TensorFlow's native `model.save()` method
- **Cache Keys**: Generated using MD5 hashing for deterministic file names
- **Data Format**: Pandas DataFrames and NumPy arrays preserved with `joblib`

## Future Enhancements

Potential improvements to the caching system:
- Time-based cache expiration
- Cache size limits with LRU eviction
- Distributed cache for multi-node training
- Cache versioning and metadata tracking
