
# Synthetic Credit Portfolio Data with VAE: Addressing Imbalance and Privacy

## 1. Introduction: The Credit Risk Analyst's Dilemma

### Story + Context + Real-World Relevance

Mr. Alex Chen, a CFA Charterholder and Senior Credit Risk Analyst at Apex Financial, is facing a critical challenge. His team is responsible for developing accurate loan default prediction models, a cornerstone of Apex's risk management strategy. However, real-world loan portfolios suffer from severe class imbalance – only 2-5% of borrowers typically default. This imbalance causes traditional models to struggle in identifying truly risky loans, often leading to models that achieve high overall accuracy but poor recall for the critical default class.

Adding to this, stringent data privacy regulations like GDPR and CCPA restrict the use and sharing of actual customer data, making it difficult to augment scarce default records or collaborate on model development across different departments or with external partners.

Alex believes generative AI, specifically Conditional Variational Autoencoders (CVAE), could offer a solution by synthesizing realistic, privacy-preserving default records. This notebook will guide Alex through applying and evaluating this advanced technique in his workflow.

### Install Required Libraries

```python
!pip install pandas numpy scikit-learn xgboost imblearn tensorflow matplotlib seaborn sdv "keras<3"
```

### Import Required Dependencies

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
    PrecisionRecallDisplay
)
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ks_2samp
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, BatchNormalization, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
# For CTGAN as an alternative, though we're implementing VAE
# from sdv.single_table import CTGAN
# from sdv.evaluation.single_table import evaluate_quality
```

## 2. Data Preparation and Understanding the Imbalance

### Story + Context + Real-World Relevance

Alex's first step is to load the raw loan data and understand its characteristics, especially the severe class imbalance. He knows that without addressing this imbalance, any predictive model will be biased towards the majority class (non-defaults). Preprocessing, such as handling missing values and feature scaling, is crucial to prepare the data for machine learning models. Finally, splitting the data into training and testing sets, ensuring stratification, will provide a reliable basis for model evaluation.

### Code cell (function definition + function execution)

```python
def load_and_preprocess_data(filepath='cs-training.csv', test_size=0.2, random_state=42):
    """
    Loads, preprocesses, and splits the credit risk dataset.

    Args:
        filepath (str): Path to the CSV dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random operations.

    Returns:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test, defaults_train, non_defaults_train, scaler
    """
    df = pd.read_csv(filepath)

    # Drop unnecessary identifier column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Rename columns for clarity as per specified dataset
    df.columns = [
        'default', 'utilization', 'age', 'past_due_30',
        'debt_ratio', 'income', 'open_credits', 'past_due_90',
        'real_estate_lines', 'past_due_60', 'dependents'
    ]

    # Handle missing values
    df['income'] = df['income'].fillna(df['income'].median())
    df['dependents'] = df['dependents'].fillna(0)
    df = df.dropna() # Drop any remaining rows with NaNs

    # Separate target variable
    X = df.drop('default', axis=1)
    y = df['default']

    # Stratified train-test split to preserve default rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrame for easier manipulation later
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


    # Separate defaults and non-defaults for targeted generation
    defaults_train = X_train_scaled[y_train == 1]
    non_defaults_train = X_train_scaled[y_train == 0]

    print(f"Total dataset size: {len(df):,}")
    print(f"Original Defaults: {df['default'].sum():,} ({df['default'].mean():.2%})")
    print(f"Original Non-defaults: {(1 - df['default']).sum():,}")
    print(f"\nTraining set size: {len(X_train_scaled):,}")
    print(f"Training set Defaults: {len(defaults_train):,} ({y_train.mean():.2%})")
    print(f"Training set Non-defaults: {len(non_defaults_train):,}")
    print(f"\nTest set size: {len(X_test_scaled):,}")
    print(f"Test set Defaults: {y_test.sum():,} ({y_test.mean():.2%})")

    return X_train_scaled, X_test_scaled, y_train, y_test, defaults_train, non_defaults_train, scaler

# Execute data loading and preprocessing
X_train_scaled, X_test_scaled, y_train, y_test, defaults_train, non_defaults_train, scaler = load_and_preprocess_data()
```

### Markdown cell (explanation of execution)

The output clearly shows the severe class imbalance in the training data, where defaults represent a very small percentage of the total loans. This imbalance is the core problem Alex needs to address, as it directly impacts the model's ability to learn the characteristics of defaulting borrowers effectively. The data has been scaled, which is crucial for many machine learning algorithms, including the upcoming VAE.

## 3. Establishing a Baseline: XGBoost on Raw Data

### Story + Context + Real-World Relevance

Before applying any augmentation techniques, Alex needs to establish a performance benchmark. He will train an XGBoost model, a powerful and commonly used algorithm in finance, directly on the imbalanced, preprocessed training data. Evaluating this "baseline" model using metrics like Area Under the Receiver Operating Characteristic Curve (AUC) and Average Precision (AP) will provide a starting point for measuring the impact of subsequent data augmentation strategies. For imbalanced datasets, AUC and AP are more informative than simple accuracy, as they capture the model's ability to distinguish between classes across various thresholds.

### Code cell (function definition + function execution)

```python
def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, model_name="Baseline"):
    """
    Trains an XGBoost classifier and evaluates its performance.

    Args:
        X_train (pd.DataFrame or np.array): Training features.
        y_train (pd.Series or np.array): Training labels.
        X_test (pd.DataFrame or np.array): Test features.
        y_test (pd.Series or np.array): Test labels.
        model_name (str): Name of the model for reporting.

    Returns:
        tuple: Trained model, y_probabilities, AUC, Average Precision
    """
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='auc', # Evaluation metric used for early stopping
        random_state=42,
        use_label_encoder=False # Suppress warning for older versions
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    print(f"{model_name} Model Performance:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print("-" * 30)

    return model, y_prob, auc, ap

# Execute baseline model training and evaluation
model_base, y_prob_base, auc_base, ap_base = train_and_evaluate_xgboost(
    X_train_scaled, y_train, X_test_scaled, y_test, "Baseline (No Augmentation)"
)
```

### Markdown cell (explanation of execution)

The baseline XGBoost model provides the initial performance metrics. Alex notes the AUC and Average Precision scores. These values will serve as the crucial benchmark to compare against models trained with data augmentation, helping him quantify the value added by more sophisticated techniques.

## 4. Classical Augmentation: SMOTE and its Limitations

### Story + Context + Real-World Relevance

Alex understands that traditional oversampling methods like SMOTE (Synthetic Minority Over-sampling Technique) are a common first approach to address class imbalance. SMOTE works by creating synthetic samples for the minority class along the line segments connecting existing minority class instances. While straightforward, Alex is aware that SMOTE's linear interpolation approach might not capture complex, non-linear dependencies in the data, and it offers no privacy benefits as synthetic samples are directly derived from real ones.

**Mathematical Formulation for SMOTE:**

SMOTE generates a new point $ \tilde{x} $ by linearly interpolating between a minority sample $ x_i $ and one of its $k$-nearest minority neighbors $ x_j $. The formula is:

$$
\tilde{x} = x_i + \lambda(x_j - x_i)
$$

where $ \lambda $ is a random number between 0 and 1 (i.e., $ \lambda \sim U(0, 1) $).

This means all SMOTE-generated points lie on the line segments between existing real data points. In geometric terms, they stay within the "convex hull" of the minority class data, potentially limiting the diversity and realism for complex datasets.

### Code cell (function definition + function execution)

```python
def apply_smote_and_train_xgboost(X_train, y_train, X_test, y_test, random_state=42):
    """
    Applies SMOTE to training data, then trains and evaluates an XGBoost classifier.

    Args:
        X_train (pd.DataFrame or np.array): Original training features.
        y_train (pd.Series or np.array): Original training labels.
        X_test (pd.DataFrame or np.array): Test features.
        y_test (pd.Series or np.array): Test labels.
        random_state (int): Seed for random operations.

    Returns:
        tuple: Trained model, y_probabilities, AUC, Average Precision, X_smote, y_smote
    """
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    print(f"Training set after SMOTE: {len(X_smote):,} samples")
    print(f"  Defaults: {y_smote.sum():,} ({y_smote.mean():.2%})")
    print(f"  Non-defaults: {(len(X_smote) - y_smote.sum()):,}")
    print("-" * 30)

    model_smote, y_prob_smote, auc_smote, ap_smote = train_and_evaluate_xgboost(
        X_smote, y_smote, X_test, y_test, "SMOTE Augmented"
    )

    return model_smote, y_prob_smote, auc_smote, ap_smote, X_smote, y_smote

# Execute SMOTE augmentation and model training
model_smote, y_prob_smote, auc_smote, ap_smote, X_smote_resampled, y_smote_resampled = apply_smote_and_train_xgboost(
    X_train_scaled, y_train, X_test_scaled, y_test
)
```

### Markdown cell (explanation of execution)

Alex observes that SMOTE successfully balances the training set, and the model's performance (AUC, AP) shows an improvement over the baseline. However, he remains skeptical about SMOTE's ability to generate truly novel and diverse samples that capture the intricate, non-linear relationships often present in financial data. Furthermore, the privacy aspect is entirely unaddressed by SMOTE, as its synthetic samples are essentially "mixtures" of real ones, offering no true anonymity.

## 5. Advanced Augmentation: Building and Training a Conditional VAE

### Story + Context + Real-World Relevance

To overcome the limitations of SMOTE, particularly regarding capturing complex data distributions and preserving privacy, Alex turns to a more sophisticated generative AI technique: the Conditional Variational Autoencoder (CVAE). A CVAE learns a compressed, probabilistic representation (latent space) of the input data, conditioned on a specific label (in this case, default/non-default). This conditioning allows Alex to generate synthetic data for a *specific class*, like defaults, providing targeted augmentation. The CVAE's ability to sample from this learned latent distribution allows it to create entirely new data points that are statistically consistent with the real data, even capturing non-linear dependencies. Unlike SMOTE, these synthetic records do not directly correspond to any real individual, offering a degree of privacy preservation.

**Mathematical Formulation for CVAE:**

The Conditional VAE aims to learn a mapping from a latent space $ \mathbf{z} $ to the data space $ \mathbf{x} $, conditioned on a class label $ \mathbf{y} $. It consists of an encoder and a decoder.

1.  **Encoder (Recognition Model):** $ q_{\phi}(\mathbf{z}|\mathbf{x}, \mathbf{y}) $
    This part maps the input data $ \mathbf{x} $ and its condition $ \mathbf{y} $ to a probabilistic distribution in the latent space $ \mathbf{z} $. It is typically parameterized as a Gaussian distribution:
    $$
    q_{\phi}(\mathbf{z}|\mathbf{x}, \mathbf{y}) = \mathcal{N}(\mu_{\phi}(\mathbf{x}, \mathbf{y}), \text{diag}(\sigma^2_{\phi}(\mathbf{x}, \mathbf{y})))
    $$
    where $ \mu_{\phi} $ and $ \sigma^2_{\phi} $ are neural networks with parameters $ \phi $.

2.  **Decoder (Generative Model):** $ p_{\theta}(\mathbf{x}|\mathbf{z}, \mathbf{y}) $
    This part reconstructs the data $ \mathbf{x} $ from the latent representation $ \mathbf{z} $ and the condition $ \mathbf{y} $. It also uses neural networks with parameters $ \theta $.

The CVAE is trained to optimize a loss function that balances two objectives:
*   **Reconstruction Loss:** Ensures the decoder can accurately reconstruct the input data from its latent representation.
*   **KL Divergence Regularization:** Forces the latent distribution $ q_{\phi}(\mathbf{z}|\mathbf{x}, \mathbf{y}) $ to be close to a prior distribution (typically a standard normal distribution $ p(\mathbf{z}) = \mathcal{N}(0, I) $), preventing the latent space from becoming too complex or irregular.

The overall loss function (Evidence Lower Bound - ELBO) is:
$$
\mathcal{L}(\phi, \theta; \mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \hat{\mathbf{x}}||^2 + \beta D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}, \mathbf{y})||p(\mathbf{z}))
$$
where $ ||\mathbf{x} - \hat{\mathbf{x}}||^2 $ is the reconstruction loss (e.g., Mean Squared Error for continuous data), $ \beta $ is a regularization parameter (often 1), and $ D_{KL} $ is the Kullback-Leibler divergence. The $ \beta $ term here specifically relates to how strongly the latent distribution is regularized towards a normal prior.

### Code cell (function definition + function execution)

```python
def build_cvae(input_dim, latent_dim=8, n_classes=2, hidden_dims=[64, 32]):
    """
    Builds and compiles a Conditional Variational Autoencoder (CVAE) model.

    Args:
        input_dim (int): Number of features in the input data.
        latent_dim (int): Dimensionality of the latent space.
        n_classes (int): Number of classes for conditioning.
        hidden_dims (list): List of integers for hidden layer dimensions.

    Returns:
        tuple: cvae_model, encoder_model, decoder_model
    """
    # ------------------ Encoder ------------------
    x_input = Input(shape=(input_dim,), name='x_input')
    y_input = Input(shape=(n_classes,), name='y_input') # One-hot encoded label

    # Concatenate x and y for conditional input to encoder
    enc_in = Concatenate()([x_input, y_input])

    h = enc_in
    for dim in hidden_dims:
        h = Dense(dim, activation='relu')(h)
        h = BatchNormalization()(h)
        h = Dropout(0.2)(h)

    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)

    # Sampling function (reparameterization trick)
    def sampling(args):
        z_mean, z_log_var = args
        eps = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * eps

    z = Lambda(sampling, name='z')([z_mean, z_log_var])

    encoder = Model([x_input, y_input], [z_mean, z_log_var, z], name='encoder')

    # ------------------ Decoder ------------------
    z_input = Input(shape=(latent_dim,), name='z_input')
    y_dec_input = Input(shape=(n_classes,), name='y_dec_input')

    # Concatenate z and y for conditional input to decoder
    dec_in = Concatenate()([z_input, y_dec_input])

    h_dec = dec_in
    for dim in reversed(hidden_dims): # Mirror encoder architecture
        h_dec = Dense(dim, activation='relu')(h_dec)
        h_dec = BatchNormalization()(h_dec)

    x_decoded = Dense(input_dim, activation='linear', name='x_decoded_output')(h_dec) # Output is raw feature values

    decoder = Model([z_input, y_dec_input], x_decoded, name='decoder')

    # ------------------ Full CVAE Model ------------------
    # Connect encoder and decoder
    z_mean_out, z_log_var_out, z_out = encoder([x_input, y_input])
    x_recon = decoder([z_out, y_input])

    cvae = Model([x_input, y_input], x_recon, name='cvae')

    # CVAE loss: reconstruction loss + KL divergence loss
    # Reconstruction loss (MSE)
    recon_loss = K.mean(K.square(x_input - x_recon), axis=-1)

    # KL divergence loss
    kl_loss = -0.5 * K.mean(
        1 + z_log_var_out - K.square(z_mean_out) - K.exp(z_log_var_out),
        axis=-1
    )

    cvae.add_loss(K.mean(recon_loss + 0.5 * kl_loss)) # Beta = 0.5 as an example

    cvae.compile(optimizer='adam')
    return cvae, encoder, decoder

# Prepare one-hot labels for training
y_train_oh = to_categorical(y_train, num_classes=2)

# Get input dimension from scaled training data
input_dim = X_train_scaled.shape[1]

# Build CVAE
latent_dim_cvae = 8
cvae, encoder, decoder = build_cvae(input_dim, latent_dim=latent_dim_cvae, n_classes=2)

print("CVAE Model Summary:")
cvae.summary()

# Define EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity
    verbose=1
)

# Train the CVAE
print("\nTraining Conditional VAE...")
cvae.fit(
    [X_train_scaled, y_train_oh],
    epochs=100,
    batch_size=64,
    validation_split=0.15, # Use 15% of training data for validation during CVAE training
    callbacks=[early_stopping],
    verbose=0 # Set to 1 to see progress bar, 0 for silent
)
print("CVAE trained successfully.")
```

### Markdown cell (explanation of execution)

The CVAE has been built and trained. Alex understands that the training process involved minimizing a combined loss, ensuring both accurate reconstruction of input data and a well-behaved latent space. The model summary provides an overview of the CVAE's architecture. Now, the trained `decoder` component of the CVAE can be used to generate entirely new synthetic data points, conditioned to be specifically for the default class. This is a crucial step towards balancing the dataset in a privacy-preserving manner.

## 6. VAE-Augmented Modeling and Initial Performance Check

### Story + Context + Real-World Relevance

With the CVAE trained, Alex can now leverage its generative capabilities. He will use the CVAE's decoder to create a targeted number of synthetic default records, aiming to balance the original training dataset. This augmented dataset will then be used to train another XGBoost model. By comparing its performance to the baseline and SMOTE models, Alex can get a preliminary understanding of the CVAE's impact on predictive accuracy. A key consideration here, derived from financial best practices and research (e.g., CFA Synthetic Data report, Tait, 2025), is the `more data is not always better` principle. Over-generating synthetic data can introduce noise, so Alex will cap the synthetic defaults to a reasonable multiple of the existing real defaults (e.g., 5x) to maintain signal integrity.

### Code cell (function definition + function execution)

```python
def generate_synthetic_defaults(decoder_model, n_samples, latent_dim, n_classes=2):
    """
    Generates synthetic default borrower records using the CVAE decoder.

    Args:
        decoder_model (tf.keras.Model): The trained CVAE decoder model.
        n_samples (int): Number of synthetic samples to generate.
        latent_dim (int): Dimensionality of the latent space.
        n_classes (int): Number of classes used for conditioning (e.g., 2 for default/non-default).

    Returns:
        pd.DataFrame: DataFrame of generated synthetic default records.
    """
    # Sample from a standard normal distribution for the latent space
    z_samples = np.random.normal(0, 1, (n_samples, latent_dim))

    # Condition for default class (one-hot encoded [0, 1])
    y_default_cond = np.zeros((n_samples, n_classes))
    y_default_cond[:, 1] = 1.0 # Set the second element to 1 for default class

    # Generate synthetic records using the decoder
    synthetic_records_np = decoder_model.predict([z_samples, y_default_cond], verbose=0)
    synthetic_records_df = pd.DataFrame(synthetic_records_np, columns=X_train_scaled.columns)
    return synthetic_records_df

def augment_with_vae_and_train_xgboost(
    X_train_scaled, y_train, X_test_scaled, y_test,
    defaults_train_df, non_defaults_train_df, decoder_model, latent_dim_cvae,
    max_synthetic_ratio=5
):
    """
    Generates synthetic defaults using VAE, augments training data, and trains an XGBoost model.

    Args:
        X_train_scaled (pd.DataFrame): Scaled real training features.
        y_train (pd.Series): Real training labels.
        X_test_scaled (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test labels.
        defaults_train_df (pd.DataFrame): Real default training samples.
        non_defaults_train_df (pd.DataFrame): Real non-default training samples.
        decoder_model (tf.keras.Model): Trained VAE decoder.
        latent_dim_cvae (int): Latent dimension of the CVAE.
        max_synthetic_ratio (int): Max ratio of synthetic to real defaults to generate.

    Returns:
        tuple: Trained model, y_probabilities, AUC, Average Precision, X_augmented, y_augmented
    """
    # Determine number of synthetic defaults needed to balance the dataset
    n_defaults_real = len(defaults_train_df)
    n_non_defaults_real = len(non_defaults_train_df)
    n_defaults_needed = n_non_defaults_real - n_defaults_real

    # Cap the number of synthetic defaults to avoid overwhelming real signal
    n_synthetic_to_generate = min(n_defaults_needed, n_defaults_real * max_synthetic_ratio)

    if n_synthetic_to_generate <= 0:
        print("No synthetic defaults needed or cap reached. Skipping VAE augmentation.")
        return None, None, None, None, X_train_scaled, y_train

    synth_defaults_df = generate_synthetic_defaults(decoder_model, n_synthetic_to_generate, latent_dim_cvae)
    print(f"Generated {len(synth_defaults_df):,} synthetic default records.")

    # Augment the training set
    X_augmented = pd.concat([X_train_scaled, synth_defaults_df], ignore_index=True)
    y_augmented = pd.concat([y_train, pd.Series(1, index=range(len(synth_defaults_df)))], ignore_index=True)

    # Shuffle the augmented data
    shuffled_indices = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented.iloc[shuffled_indices].reset_index(drop=True)
    y_augmented = y_augmented.iloc[shuffled_indices].reset_index(drop=True)

    print(f"Augmented training set size: {len(X_augmented):,}")
    print(f"  Defaults: {y_augmented.sum():,} ({y_augmented.mean():.2%})")
    print(f"  Non-defaults: {(len(X_augmented) - y_augmented.sum()):,}")
    print("-" * 30)

    model_vae, y_prob_vae, auc_vae, ap_vae = train_and_evaluate_xgboost(
        X_augmented, y_augmented, X_test_scaled, y_test, "VAE Augmented"
    )

    return model_vae, y_prob_vae, auc_vae, ap_vae, X_augmented, y_augmented

# Execute VAE augmentation and model training
model_vae, y_prob_vae, auc_vae, ap_vae, X_vae_augmented, y_vae_augmented = augment_with_vae_and_train_xgboost(
    X_train_scaled, y_train, X_test_scaled, y_test,
    defaults_train, non_defaults_train, decoder, latent_dim_cvae,
    max_synthetic_ratio=5 # Cap synthetic defaults to 5x real defaults
)
```

### Markdown cell (explanation of execution)

Alex observes the performance of the XGBoost model trained on the VAE-augmented dataset. He compares the AUC and Average Precision with the baseline and SMOTE models. While improvements are often seen, Alex knows that simply looking at performance metrics is not enough. He needs to rigorously assess the *quality*, *utility*, and *privacy* of the synthetic data itself before making a recommendation to Apex Financial's risk committee.

## 7. Comprehensive Synthetic Data Assessment: Quality, Utility, and Privacy

### Story + Context + Real-World Relevance

For Apex Financial to adopt synthetic data, Alex must provide a thorough assessment beyond just model performance. He needs to quantify if the synthetic data accurately reflects the real data's characteristics (quality), if models trained on synthetic data generalize well to real data (utility), and crucially, if the synthetic data truly preserves privacy (privacy). This section demonstrates the gold standard tests for these critical aspects, addressing concerns from both technical and regulatory perspectives.

### Code cell (function definition + function execution)

```python
# Inverse transform for interpretable comparison (for plotting and KS tests)
synth_defaults_raw = scaler.inverse_transform(synth_defaults_df)
synth_defaults_inv_df = pd.DataFrame(synth_defaults_raw, columns=X_train_scaled.columns)

real_defaults_raw = scaler.inverse_transform(defaults_train)
real_defaults_inv_df = pd.DataFrame(real_defaults_raw, columns=X_train_scaled.columns)

# Separate SMOTE-generated records for privacy comparison
# SMOTE-generated records start after the original training data in X_smote_resampled
X_smote_new_scaled = X_smote_resampled[len(X_train_scaled):]


# --- 7.1 Train-on-Synthetic, Test-on-Real (TSTR) Protocol ---
print("--- 7.1 Train-on-Synthetic, Test-on-Real (TSTR) Protocol ---")

# Generate a balanced synthetic dataset for TSTR (both classes)
# Generate synthetic non-defaults
n_synthetic_non_defaults = len(defaults_train) - len(non_defaults_train) if len(defaults_train) > len(non_defaults_train) else 0 # Example, can be more complex
if n_synthetic_non_defaults > 0:
    z_nd = np.random.normal(0, 1, (n_synthetic_non_defaults, latent_dim_cvae))
    y_nd_cond = np.zeros((n_synthetic_non_defaults, 2))
    y_nd_cond[:, 0] = 1.0 # Condition for non-default class
    synth_non_defaults_df = pd.DataFrame(decoder.predict([z_nd, y_nd_cond], verbose=0), columns=X_train_scaled.columns)
else:
    synth_non_defaults_df = pd.DataFrame(columns=X_train_scaled.columns)

# Combine real non-defaults with synthetic non-defaults if needed, or just use real non-defaults if sufficient
# For a balanced *synthetic-only* dataset, we usually generate from both classes to match counts
# Let's generate a full synthetic dataset of the size of the original training set, balanced.
target_size_per_class = max(len(defaults_train), len(non_defaults_train)) # Roughly aiming for this, or just target half the original training set size for each class
n_synthetic_per_class_tstr = len(non_defaults_train) # To match the majority class count

if n_synthetic_per_class_tstr <= 0:
    print("Not enough samples to generate for TSTR balanced synthetic dataset. Skipping TSTR.")
    auc_tstr = np.nan
else:
    synth_defaults_tstr_df = generate_synthetic_defaults(decoder, n_synthetic_per_class_tstr, latent_dim_cvae)

    z_non_default_tstr = np.random.normal(0, 1, (n_synthetic_per_class_tstr, latent_dim_cvae))
    y_non_default_cond_tstr = np.zeros((n_synthetic_per_class_tstr, 2))
    y_non_default_cond_tstr[:, 0] = 1.0 # Condition for non-default
    synth_non_defaults_tstr_df = pd.DataFrame(decoder.predict([z_non_default_tstr, y_non_default_cond_tstr], verbose=0), columns=X_train_scaled.columns)

    X_synth_only = pd.concat([synth_defaults_tstr_df, synth_non_defaults_tstr_df], ignore_index=True)
    y_synth_only = pd.Series([1]*len(synth_defaults_tstr_df) + [0]*len(synth_non_defaults_tstr_df))

    # Shuffle the synthetic-only dataset
    shuffled_idx_tstr = np.random.permutation(len(X_synth_only))
    X_synth_only = X_synth_only.iloc[shuffled_idx_tstr].reset_index(drop=True)
    y_synth_only = y_synth_only.iloc[shuffled_idx_tstr].reset_index(drop=True)

    print(f"Training a model *only* on {len(X_synth_only):,} synthetic records for TSTR.")
    model_tstr = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        eval_metric='auc', random_state=42, use_label_encoder=False
    )
    model_tstr.fit(X_synth_only, y_synth_only)
    y_prob_tstr = model_tstr.predict_proba(X_test_scaled)[:, 1]
    auc_tstr = roc_auc_score(y_test, y_prob_tstr)
    print(f"\nTSTR AUC (trained ONLY on synthetic, tested on real): {auc_tstr:.4f}")
    print(f"Real data AUC (baseline): {auc_base:.4f}")
    if auc_base > 0:
        tstr_ratio = auc_tstr / auc_base
        print(f"TSTR Ratio: {tstr_ratio:.2%} (Target > 80% for good utility)")
    else:
        print("Baseline AUC is zero, cannot compute TSTR ratio.")


# --- 7.2 Feature Distribution Comparison (KS Test & Overlays) ---
print("\n--- 7.2 Feature Distribution Comparison (KS Test & Overlays) ---")
print("Column-wise KS Tests (Real vs Synthetic Defaults):")
ks_test_results = []
for col in X_train_scaled.columns:
    ks_stat, p_val = ks_2samp(real_defaults_inv_df[col], synth_defaults_inv_df[col])
    status = "OK" if p_val > 0.05 else "MISMATCH" # Common threshold for statistical significance
    ks_test_results.append({'Feature': col, 'KS Statistic': ks_stat, 'P-value': p_val, 'Status': status})
    print(f"  {col:20s}: KS={ks_stat:.4f}, p={p_val:.4f} [{status}]")

# Plot overlays for key features
key_features_for_overlay = ['utilization', 'income', 'debt_ratio', 'past_due_30']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(key_features_for_overlay):
    sns.histplot(real_defaults_inv_df[col], kde=True, color='blue', label='Real Defaults', ax=axes[i], stat='density', alpha=0.6)
    sns.histplot(synth_defaults_inv_df[col], kde=True, color='red', label='Synthetic Defaults (VAE)', ax=axes[i], stat='density', alpha=0.6)
    axes[i].set_title(f'Distribution of {col}', fontsize=14)
    axes[i].legend()
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Density', fontsize=12)
plt.suptitle('Marginal Feature Distribution Comparison (Real vs. Synthetic Defaults)', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig('feature_distribution_comparison.png', dpi=150)
plt.show()


# --- 7.3 Correlation Structure Preservation (Frobenius Error & Heatmaps) ---
print("\n--- 7.3 Correlation Structure Preservation (Frobenius Error & Heatmaps) ---")
real_corr = real_defaults_inv_df.corr()
synth_corr = synth_defaults_inv_df.corr()

# Calculate Frobenius Norm for correlation matrix difference
# Mathematical Formulation for Correlation Matrix Frobenius Error:
# To quantify how well the VAE preserves feature correlations:
# $$ \epsilon_{corr} = ||C_{real} - C_{synth}||_F = \sqrt{\sum_{i,j} (C_{real_{ij}} - C_{synth_{ij}})^2} $$
# where $ C $ is the Pearson correlation matrix.
frobenius_error = np.linalg.norm(real_corr - synth_corr, 'fro')
print(f"Correlation Matrix Frobenius Error (Real vs Synthetic Defaults): {frobenius_error:.4f}")
print("Target: < 0.5 for 10-feature credit data (lower is better)")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0], cbar_kws={'shrink': 0.8})
axes[0].set_title('Real Defaults: Correlation Matrix', fontsize=16)
sns.heatmap(synth_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1], cbar_kws={'shrink': 0.8})
axes[1].set_title('Synthetic Defaults (VAE): Correlation Matrix', fontsize=16)
plt.suptitle('Correlation Preservation: Real vs. Synthetic Defaults', fontsize=18, y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig('correlation_preservation.png', dpi=150)
plt.show()


# --- 7.4 Privacy Assessment (Nearest-Neighbor Distance) ---
print("\n--- 7.4 Privacy Assessment (Nearest-Neighbor Distance) ---")

# For VAE-generated synthetic defaults
nn_vae = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn_vae.fit(X_train_scaled[y_train == 1]) # Fit to real default records
distances_vae, _ = nn_vae.kneighbors(synth_defaults_df[:1000]) # Check distances for first 1000 VAE samples
min_distances_vae = distances_vae.ravel()

print("Privacy Assessment (VAE synthetic defaults):")
print(f"  Mean distance to nearest real (VAE): {np.mean(min_distances_vae):.4f}")
print(f"  Min distance (worst case VAE): {np.min(min_distances_vae):.4f}")
print(f"  Records with distance < 0.5 (VAE): {(min_distances_vae < 0.5).sum()} / {len(min_distances_vae)}")

# For SMOTE-generated records (if available)
if len(X_smote_new_scaled) > 0:
    nn_smote = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn_smote.fit(X_train_scaled[y_train == 1]) # Fit to real default records
    distances_smote, _ = nn_smote.kneighbors(X_smote_new_scaled[:1000]) # Check distances for first 1000 SMOTE samples
    min_distances_smote = distances_smote.ravel()

    print("\nPrivacy Assessment (SMOTE generated records):")
    print(f"  Mean distance to nearest real (SMOTE): {np.mean(min_distances_smote):.4f}")
    print(f"  Min distance (worst case SMOTE): {np.min(min_distances_smote):.4f}")
    print(f"  Records with distance < 0.5 (SMOTE): {(min_distances_smote < 0.5).sum()} / {len(min_distances_smote)}")
    print("  (SMOTE samples are generally closer due to linear interpolation)")

    # Plot histogram of distances
    plt.figure(figsize=(10, 6))
    sns.histplot(min_distances_vae, kde=True, color='red', label='VAE Synthetic', stat='density', alpha=0.6)
    sns.histplot(min_distances_smote, kde=True, color='blue', label='SMOTE Synthetic', stat='density', alpha=0.6)
    plt.title('Nearest-Neighbor Distance to Real Defaults (Privacy Assessment)', fontsize=16)
    plt.xlabel('Distance to Closest Real Record', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('privacy_distance_histogram.png', dpi=150)
    plt.show()
else:
    print("\nSMOTE-generated records not available for privacy comparison.")


# --- 7.5 t-SNE / PCA Scatter Plots (Visualization) ---
print("\n--- 7.5 t-SNE / PCA Scatter Plots ---")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Combine real and synthetic defaults for visualization
# Ensure consistent number of samples for comparison, e.g., 1000 from each
n_viz_samples = min(len(real_defaults_inv_df), len(synth_defaults_inv_df), 2000)

real_viz = real_defaults_inv_df.sample(n=n_viz_samples, random_state=42)
synth_viz = synth_defaults_inv_df.sample(n=n_viz_samples, random_state=42)

combined_viz_data = pd.concat([real_viz, synth_viz], ignore_index=True)
combined_viz_labels = ['Real'] * n_viz_samples + ['Synthetic (VAE)'] * n_viz_samples

# Apply PCA for initial dimensionality reduction if features are many (e.g. > 50)
# Here we have few features, so t-SNE directly is fine, but PCA can precede t-SNE.
pca = PCA(n_components=min(combined_viz_data.shape[1], 5), random_state=42)
pca_results = pca.fit_transform(combined_viz_data)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate=200) # Adjust perplexity/n_iter as needed
tsne_results = tsne.fit_transform(pca_results) # Apply t-SNE on PCA reduced data

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    hue=combined_viz_labels, palette={'Real': 'blue', 'Synthetic (VAE)': 'red'},
    alpha=0.6, s=50
)
plt.title(f't-SNE Visualization of Real vs. VAE Synthetic Defaults (n={n_viz_samples} each)', fontsize=16)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.legend(title='Data Type')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('tsne_real_vs_synthetic.png', dpi=150)
plt.show()
```

### Markdown cell (explanation of execution)

This section provides Alex with a powerful toolkit for assessing synthetic data.
*   **TSTR Protocol:** The TSTR AUC and ratio indicate how well a model trained *solely* on synthetic data performs on real data. A high ratio suggests the synthetic data captures the underlying predictive patterns effectively. Alex notes if the TSTR Ratio meets the target of `> 80%` for good utility.
*   **KS Tests and Distribution Overlays:** These visuals and statistics confirm that the marginal distributions of key features in the synthetic data closely match those in the real data. Alex looks for "OK" statuses in the KS tests and visual congruence in the overlay plots to ensure the synthetic data is statistically similar.
*   **Correlation Heatmaps and Frobenius Error:** By comparing correlation matrices, Alex can see if the inter-feature dependencies are preserved. The Frobenius error provides a single quantitative measure; a low error (e.g., $ < 0.5 $ for 10 features) indicates good preservation of multivariate relationships.
*   **Nearest-Neighbor Distance Histogram:** This is critical for privacy. VAE-generated samples should be sufficiently "far" from any real data point. Alex expects VAE samples to have larger nearest-neighbor distances compared to SMOTE, which, by design, places synthetic samples directly between real ones, offering no privacy. This analysis quantifies the `privacy advantage` of VAEs.
*   **t-SNE Plot:** This visualization helps Alex understand if synthetic defaults occupy similar regions in the feature space as real defaults, indicating realism, without necessarily overlapping perfectly, which suggests privacy.

These comprehensive assessments provide the empirical evidence Alex needs to build a strong case for using CVAE-generated data at Apex Financial.

## 8. Strategic Review: Performance, Regulatory Compliance, and Ethical AI

### Story + Context + Real-World Relevance

After extensive analysis, Alex needs to synthesize his findings for Apex Financial's risk committee and other stakeholders. This means not only comparing the performance of the different models but also engaging in crucial discussions around regulatory compliance (Basel IRB, ECOA, GDPR) and ethical considerations, particularly the risk of fairness amplification. This holistic view will guide Apex in strategically integrating synthetic data into their credit modeling workflows.

### Code cell (function definition + function execution)

```python
def plot_roc_curves(y_test, y_probs, model_names):
    """
    Plots ROC curves for multiple models on a single graph.

    Args:
        y_test (pd.Series): True labels for the test set.
        y_probs (list): List of probability arrays for the positive class from each model.
        model_names (list): List of model names corresponding to y_probs.
    """
    plt.figure(figsize=(10, 8))
    for i in range(len(y_probs)):
        fpr, tpr, _ = roc_curve(y_test, y_probs[i])
        auc = roc_auc_score(y_test, y_probs[i])
        plt.plot(fpr, tpr, label=f'{model_names[i]} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=150)
    plt.show()

# Consolidate results for comparison
comparison_data = pd.DataFrame({
    'Method': ['No Augmentation (Baseline)', 'SMOTE Augmentation', 'VAE Augmentation'],
    'AUC': [auc_base, auc_smote, auc_vae],
    'Average Precision': [ap_base, ap_smote, ap_vae],
    'Privacy': ['N/A (real only)', 'Low (Interpolation Risk)', 'Moderate (Distribution Sampling)'],
    'Diversity': ['N/A', 'Low (Convex Hull)', 'High (Learned Distribution)']
})

print("--- FOUR-WAY AUGMENTATION COMPARISON ---")
print("=" * 70)
print(comparison_data.to_string(index=False))
print("=" * 70)

# Add TSTR ratio to comparison for discussion
if 'tstr_ratio' in locals():
    print(f"\nTSTR Ratio (Model trained on VAE synthetic, tested on real): {tstr_ratio:.2%}")

# Bar chart comparison of AUC and Average Precision
plt.figure(figsize=(12, 7))
bar_width = 0.35
index = np.arange(len(comparison_data['Method']))

plt.bar(index, comparison_data['AUC'], bar_width, label='AUC', color='steelblue', alpha=0.8)
plt.bar(index + bar_width, comparison_data['Average Precision'], bar_width, label='Average Precision', color='coral', alpha=0.8)

plt.xlabel('Augmentation Strategy', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Credit Default Model Performance: Augmentation Strategy Comparison', fontsize=16)
plt.xticks(index + bar_width / 2, comparison_data['Method'], rotation=15, ha='right', fontsize=10)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('augmentation_comparison_barchart.png', dpi=150)
plt.show()

# Plot ROC curves for all models
y_probabilities_list = [y_prob_base, y_prob_smote, y_prob_vae]
model_names_list = ['Baseline', 'SMOTE Augmented', 'VAE Augmented']
plot_roc_curves(y_test, y_probabilities_list, model_names_list)
```

### Markdown cell (explanation of execution)

Alex's presentation to Apex Financial's risk committee is now complete with concrete evidence.
The comparative bar charts and ROC curves visually demonstrate the uplift in model performance (AUC, Average Precision) achieved through VAE augmentation, surpassing SMOTE and significantly outperforming the baseline. The detailed assessment of synthetic data quality, utility, and privacy from the previous section underpins these performance gains.

**Discussion Points for Financial Professionals (Alex's key takeaways):**

1.  **Performance & Utility:** The VAE-augmented model generally shows better AUC and Average Precision, indicating a more robust ability to identify defaults. The TSTR ratio confirms that models trained on VAE synthetic data retain substantial predictive power on real data, validating its utility.
2.  **Privacy Preservation:** VAE-generated data offers a significant privacy advantage over SMOTE, as evidenced by the greater nearest-neighbor distances. This is crucial for compliance with regulations like **GDPR** and **CCPA**, enabling data sharing for model validation, stress testing (e.g., under **Basel IRB** framework), and research without exposing sensitive customer information.
3.  **Data Quality:** The KS tests and correlation matrix comparisons demonstrate that VAE successfully preserves the statistical properties and inter-feature relationships of the real data, which is vital for building trustworthy models.
4.  **Regulatory Scrutiny:** While synthetic data is increasingly accepted, regulators (e.g., OCC, PRA, ECB) still primarily require models to be developed on real data. Synthetic data is best viewed as a *supplement* for augmentation, validation, and collaboration. Apex Financial needs clear policies on how synthetic data is generated, validated, and used, maintaining rigorous oversight of its quality.
5.  **Ethical Considerations (Fairness Amplification Risk):** Alex warns the committee about the "fairness amplification risk." If the original real training data contains historical biases (e.g., discriminatory lending practices impacting certain demographic groups), a CVAE will faithfully reproduce, and potentially amplify, these biases in the synthetic data. This could lead to models perpetuating unfair outcomes. Therefore, rigorous bias audits on both real and synthetic data, and the resulting models, are imperative, aligning with principles of **ECOA (Equal Credit Opportunity Act)**.
6.  **Production Readiness:** For production deployments, libraries like `sdv` offering models like CTGAN could be explored as they handle mixed data types and offer robust implementations out-of-the-box, potentially reducing custom code complexity.

Alex concludes that Conditional VAEs are a powerful tool for Apex Financial, offering a principled approach to tackle class imbalance and privacy concerns simultaneously, thereby enhancing risk management capabilities and accelerating model development within a responsible AI framework.

