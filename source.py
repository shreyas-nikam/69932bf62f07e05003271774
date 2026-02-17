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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Conditional import for tf.keras.ops for Keras 3 (TF 2.16+) compatibility
try:
    import keras
    tf.keras.ops = keras.ops
except ImportError:
    # Fallback for older TensorFlow versions or other setups
    # In such cases, tf.math functions might be directly used or K. functions.
    pass


def load_and_preprocess_data(filepath=None, test_size=0.2, random_state=42, n_samples=100000):
    """
    Generates synthetic data, preprocesses it, and splits it into training and test sets.

    Args:
        filepath (str): Path to the CSV dataset. (Ignored for synthetic data generation)
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random operations.
        n_samples (int): Number of synthetic samples to generate.

    Returns:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test, defaults_train, non_defaults_train, scaler
    """
    np.random.seed(random_state)
    columns = [
        'default', 'utilization', 'age', 'past_due_30',
        'debt_ratio', 'income', 'open_credits', 'past_due_90',
        'real_estate_lines', 'past_due_60', 'dependents'
    ]
    df = pd.DataFrame(index=range(n_samples), columns=columns)

    df['age'] = np.random.randint(21, 90, n_samples)
    df['income'] = np.random.lognormal(mean=9.5, sigma=1.0, size=n_samples)
    df['utilization'] = np.random.beta(a=0.5, b=5, size=n_samples) * 1.5
    df['debt_ratio'] = np.random.beta(a=0.5, b=5, size=n_samples) * 5
    df['open_credits'] = np.random.randint(0, 20, n_samples)
    df['real_estate_lines'] = np.random.randint(0, 10, n_samples)
    df['dependents'] = np.random.poisson(lam=0.5, size=n_samples)

    df['past_due_30'] = (np.random.rand(n_samples) < 0.9).astype(int) * np.random.poisson(lam=0.1, size=n_samples)
    df['past_due_60'] = (df['past_due_30'] > 0).astype(int) * (np.random.rand(n_samples) < 0.7).astype(int) * np.random.poisson(lam=0.05, size=n_samples)
    df['past_due_90'] = (df['past_due_60'] > 0).astype(int) * (np.random.rand(n_samples) < 0.5).astype(int) * np.random.poisson(lam=0.02, size=n_samples)

    default_prob = (0.01
                    + 0.05 * df['utilization']
                    + 0.005 * (90 - df['age']) / 70
                    + 0.02 * (1 / (df['income'] + 1000) * 1000)
                    + 0.03 * df['debt_ratio']
                    + 0.05 * (df['past_due_30'] > 0).astype(int)
                    + 0.08 * (df['past_due_60'] > 0).astype(int)
                    + 0.10 * (df['past_due_90'] > 0).astype(int)
                   ).clip(0, 1)

    df['default'] = (np.random.rand(n_samples) < default_prob).astype(int)

    current_default_rate = df['default'].mean()
    if current_default_rate < 0.02 or current_default_rate > 0.05:
        print(f"Initial synthetic default rate: {current_default_rate:.2%}. Adjusting...")
        num_defaults_to_flip = int(n_samples * 0.03 - df['default'].sum())
        if num_defaults_to_flip > 0:
            non_default_indices = df[df['default'] == 0].index.to_numpy()
            np.random.shuffle(non_default_indices)
            df.loc[non_default_indices[:num_defaults_to_flip], 'default'] = 1
        elif num_defaults_to_flip < 0:
            default_indices = df[df['default'] == 1].index.to_numpy()
            np.random.shuffle(default_indices)
            df.loc[default_indices[:abs(num_defaults_to_flip)], 'default'] = 0

    df['income'] = df['income'].fillna(df['income'].median())
    df['dependents'] = df['dependents'].fillna(0)
    df = df.dropna()

    X = df.drop('default', axis=1)
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns).reset_index(drop=True)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    defaults_train = X_train_scaled[y_train == 1]
    non_defaults_train = X_train_scaled[y_train == 0]

    print(f"Total synthetic dataset size: {len(df):,}")
    print(f"Synthetic Defaults: {df['default'].sum():,} ({df['default'].mean():.2%})")
    print(f"Synthetic Non-defaults: {(1 - df['default']).sum():,}")
    print(f"\nTraining set size: {len(X_train_scaled):,}")
    print(f"Training set Defaults: {len(defaults_train):,} ({y_train.mean():.2%})")
    print(f"Training set Non-defaults: {len(non_defaults_train):,}")
    print(f"\nTest set size: {len(X_test_scaled):,}")
    print(f"Test set Defaults: {y_test.sum():,} ({y_test.mean():.2%})")

    return X_train_scaled, X_test_scaled, y_train, y_test, defaults_train, non_defaults_train, scaler

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
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False
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

class CVAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=8, n_classes=2, hidden_dims=[64, 32], **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims

        self.encoder_layers = []
        for dim in hidden_dims:
            self.encoder_layers.append(tf.keras.layers.Dense(dim, activation='relu'))
            self.encoder_layers.append(tf.keras.layers.BatchNormalization())
            self.encoder_layers.append(tf.keras.layers.Dropout(0.2))
        self.z_mean_layer = tf.keras.layers.Dense(latent_dim, name='z_mean_output')
        self.z_log_var_layer = tf.keras.layers.Dense(latent_dim, name='z_log_var_output')

        self.decoder_layers = []
        for dim in reversed(hidden_dims):
            self.decoder_layers.append(tf.keras.layers.Dense(dim, activation='relu'))
            self.decoder_layers.append(tf.keras.layers.BatchNormalization())
        self.decoder_output_layer = tf.keras.layers.Dense(input_dim, activation='linear', name='x_decoded_output')

        x_input_enc = tf.keras.layers.Input(shape=(input_dim,), name='x_input')
        y_input_enc = tf.keras.layers.Input(shape=(n_classes,), name='y_input')
        enc_in = tf.keras.layers.Concatenate()([x_input_enc, y_input_enc])
        h_enc = enc_in
        for layer in self.encoder_layers:
            h_enc = layer(h_enc)
        z_mean_enc = self.z_mean_layer(h_enc)
        z_log_var_enc = self.z_log_var_layer(h_enc)

        def sampling(args):
            z_mean, z_log_var = args
            eps = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.keras.ops.exp(0.5 * z_log_var) * eps
        z_enc = tf.keras.layers.Lambda(sampling, name='z_sample')([z_mean_enc, z_log_var_enc])
        self.encoder = tf.keras.models.Model(inputs=[x_input_enc, y_input_enc], outputs=[z_mean_enc, z_log_var_enc, z_enc], name='encoder')

        z_input_dec = tf.keras.layers.Input(shape=(latent_dim,), name='z_input')
        y_input_dec = tf.keras.layers.Input(shape=(n_classes,), name='y_input_dec')
        dec_in = tf.keras.layers.Concatenate()([z_input_dec, y_input_dec])
        h_dec = dec_in
        for layer in self.decoder_layers:
            h_dec = layer(h_dec)
        x_decoded_dec = self.decoder_output_layer(h_dec)
        self.decoder = tf.keras.models.Model(inputs=[z_input_dec, y_input_dec], outputs=x_decoded_dec, name='decoder')


    def call(self, inputs):
        x_input, y_input = inputs
        z_mean_out, z_log_var_out, z_out = self.encoder([x_input, y_input])
        x_recon = self.decoder([z_out, y_input])

        recon_loss = tf.keras.ops.mean(tf.keras.ops.square(x_input - x_recon), axis=-1)
        kl_loss = -0.5 * tf.keras.ops.mean(
            1 + z_log_var_out - tf.keras.ops.square(z_mean_out) - tf.keras.ops.exp(z_log_var_out),
            axis=-1
        )
        total_loss = tf.keras.ops.mean(recon_loss + 0.5 * kl_loss)

        self.add_loss(total_loss)

        return x_recon

def train_cvae(X_train_scaled, y_train, latent_dim=8, n_classes=2, epochs=100, batch_size=64, validation_split=0.15):
    """
    Trains the Conditional Variational Autoencoder (CVAE) model.

    Args:
        X_train_scaled (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training labels.
        latent_dim (int): Dimensionality of the latent space.
        n_classes (int): Number of classes for conditioning.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of training data to use for validation.

    Returns:
        tuple: Trained CVAE model instance, encoder model, decoder model, latent dimension.
    """
    y_train_oh = to_categorical(y_train, num_classes=n_classes)
    input_dim = X_train_scaled.shape[1]

    cvae_model_instance = CVAE(input_dim, latent_dim=latent_dim, n_classes=n_classes)
    cvae_model_instance.compile(optimizer='adam')

    print("CVAE Model Summary:")
    cvae_model_instance.build(input_shape=[(None, input_dim), (None, n_classes)])
    cvae_model_instance.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    print("\nTraining Conditional VAE...")
    history = cvae_model_instance.fit(
        [X_train_scaled, y_train_oh],
        X_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0
    )
    print("CVAE trained successfully.")

    encoder = cvae_model_instance.encoder
    decoder = cvae_model_instance.decoder

    return cvae_model_instance, encoder, decoder, latent_dim

def generate_synthetic_defaults(decoder_model, n_samples, latent_dim, n_classes=2):
    """
    Generates synthetic default borrower records using the CVAE decoder.

    Args:
        decoder_model (tf.keras.Model): The trained CVAE decoder model.
        n_samples (int): Number of synthetic samples to generate.
        latent_dim (int): Dimensionality of the latent space.
        n_classes (int): Number of classes used for conditioning (e.g., 2 for default/non-default).

    Returns:
        np.array: Numpy array of generated synthetic default records.
    """
    if n_samples <= 0:
        return np.empty((0, decoder_model.output_shape[1])) # Return empty array with correct feature dim

    z_samples = np.random.normal(0, 1, (n_samples, latent_dim))
    y_default_cond = np.zeros((n_samples, n_classes))
    y_default_cond[:, 1] = 1.0

    synthetic_records_np = decoder_model.predict([z_samples, y_default_cond], verbose=0)
    return synthetic_records_np


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
        tuple: Trained model, y_probabilities, AUC, Average Precision, X_augmented, y_augmented, synth_defaults_df (scaled)
    """
    n_defaults_real = len(defaults_train_df)
    n_non_defaults_real = len(non_defaults_train_df)
    n_defaults_needed = n_non_defaults_real - n_defaults_real

    n_synthetic_to_generate = min(n_defaults_needed, n_defaults_real * max_synthetic_ratio)
    n_synthetic_to_generate = max(0, int(n_synthetic_to_generate)) # Ensure non-negative integer

    synth_defaults_df = pd.DataFrame(columns=X_train_scaled.columns) # Initialize empty
    if n_synthetic_to_generate <= 0:
        print("No synthetic defaults needed or cap reached. Skipping VAE augmentation.")
        return None, None, None, None, X_train_scaled, y_train, synth_defaults_df

    synth_defaults_np = generate_synthetic_defaults(decoder_model, n_synthetic_to_generate, latent_dim_cvae)
    synth_defaults_df = pd.DataFrame(synth_defaults_np, columns=X_train_scaled.columns)
    print(f"Generated {len(synth_defaults_df):,} synthetic default records.")

    X_augmented = pd.concat([X_train_scaled, synth_defaults_df], ignore_index=True)
    y_augmented = pd.concat([y_train, pd.Series(1, index=range(len(synth_defaults_df)))], ignore_index=True)

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

    return model_vae, y_prob_vae, auc_vae, ap_vae, X_augmented, y_augmented, synth_defaults_df


def run_tstr_protocol(X_train_scaled, y_train, X_test_scaled, y_test, decoder, latent_dim_cvae, auc_base, n_classes=2, random_state=42):
    """
    Implements the Train-on-Synthetic, Test-on-Real (TSTR) protocol.

    Args:
        X_train_scaled (pd.DataFrame): Scaled real training features.
        y_train (pd.Series): Real training labels.
        X_test_scaled (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test labels.
        decoder (tf.keras.Model): Trained VAE decoder.
        latent_dim_cvae (int): Latent dimension of the CVAE.
        auc_base (float): Baseline AUC for comparison.
        n_classes (int): Number of classes.
        random_state (int): Seed for random operations.

    Returns:
        tuple: TSTR AUC, y_probabilities from TSTR model, TSTR ratio.
    """
    print("\n--- Train-on-Synthetic, Test-on-Real (TSTR) Protocol ---")
    np.random.seed(random_state)

    n_defaults_real = y_train.sum()
    n_non_defaults_real = (y_train == 0).sum()
    
    # Target size for each class in the synthetic dataset, aiming for balance
    # Can be max(n_defaults_real, n_non_defaults_real) or a fixed number
    n_synthetic_per_class_tstr = n_non_defaults_real # Match majority class count from real data

    if n_synthetic_per_class_tstr <= 0 or decoder is None:
        print("Not enough samples or decoder not available to generate for TSTR. Skipping TSTR.")
        return np.nan, None, np.nan

    synth_defaults_tstr_np = generate_synthetic_defaults(decoder, n_synthetic_per_class_tstr, latent_dim_cvae, n_classes)
    synth_defaults_tstr_df = pd.DataFrame(synth_defaults_tstr_np, columns=X_train_scaled.columns)

    z_non_default_tstr = np.random.normal(0, 1, (n_synthetic_per_class_tstr, latent_dim_cvae))
    y_non_default_cond_tstr = np.zeros((n_synthetic_per_class_tstr, n_classes))
    y_non_default_cond_tstr[:, 0] = 1.0
    synth_non_defaults_tstr_np = decoder.predict([z_non_default_tstr, y_non_default_cond_tstr], verbose=0)
    synth_non_defaults_tstr_df = pd.DataFrame(synth_non_defaults_tstr_np, columns=X_train_scaled.columns)

    X_synth_only = pd.concat([synth_defaults_tstr_df, synth_non_defaults_tstr_df], ignore_index=True)
    y_synth_only = pd.Series([1]*len(synth_defaults_tstr_df) + [0]*len(synth_non_defaults_tstr_df))

    shuffled_idx_tstr = np.random.permutation(len(X_synth_only))
    X_synth_only = X_synth_only.iloc[shuffled_idx_tstr].reset_index(drop=True)
    y_synth_only = y_synth_only.iloc[shuffled_idx_tstr].reset_index(drop=True)

    print(f"Training a model *only* on {len(X_synth_only):,} synthetic records for TSTR.")
    model_tstr = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        eval_metric='auc', random_state=random_state, use_label_encoder=False
    )
    model_tstr.fit(X_synth_only, y_synth_only)
    y_prob_tstr = model_tstr.predict_proba(X_test_scaled)[:, 1]
    auc_tstr = roc_auc_score(y_test, y_prob_tstr)

    print(f"\nTSTR AUC (trained ONLY on synthetic, tested on real): {auc_tstr:.4f}")
    print(f"Real data AUC (baseline): {auc_base:.4f}")
    tstr_ratio = np.nan
    if auc_base > 0:
        tstr_ratio = auc_tstr / auc_base
        print(f"TSTR Ratio: {tstr_ratio:.2%}" + " (Target > 80% for good utility)")
    else:
        print("Baseline AUC is zero, cannot compute TSTR ratio.")

    return auc_tstr, y_prob_tstr, tstr_ratio


def perform_distribution_comparison(real_defaults_inv_df, synth_defaults_inv_df, features_to_plot, output_filename='feature_distribution_comparison.png'):
    """
    Performs column-wise KS tests and plots marginal feature distributions.

    Args:
        real_defaults_inv_df (pd.DataFrame): Inverse-transformed real default samples.
        synth_defaults_inv_df (pd.DataFrame): Inverse-transformed synthetic default samples.
        features_to_plot (list): List of feature names to plot.
        output_filename (str): Filename to save the distribution comparison plot.

    Returns:
        pd.DataFrame: DataFrame containing KS test results.
    """
    print("\n--- Feature Distribution Comparison (KS Test & Overlays) ---")
    if real_defaults_inv_df.empty or synth_defaults_inv_df.empty:
        print("Skipping distribution comparison due to empty dataframes.")
        return pd.DataFrame(columns=['Feature', 'KS Statistic', 'P-value', 'Status'])

    print("Column-wise KS Tests (Real vs Synthetic Defaults):")
    ks_test_results = []
    for col in real_defaults_inv_df.columns:
        if col in synth_defaults_inv_df.columns:
            ks_stat, p_val = ks_2samp(real_defaults_inv_df[col], synth_defaults_inv_df[col])
            status = "OK" if p_val > 0.05 else "MISMATCH"
            ks_test_results.append({'Feature': col, 'KS Statistic': ks_stat, 'P-value': p_val, 'Status': status})
            print(f"  {col:20s}: KS={ks_stat:.4f}, p={p_val:.4f} [{status}]")
        else:
            print(f"  Warning: Feature '{col}' not found in synthetic data for KS test.")

    ks_results_df = pd.DataFrame(ks_test_results)

    plot_features = [f for f in features_to_plot if f in real_defaults_inv_df.columns and f in synth_defaults_inv_df.columns]
    if plot_features:
        num_plots = len(plot_features)
        fig_rows = int(np.ceil(num_plots / 2))
        fig, axes = plt.subplots(fig_rows, 2, figsize=(16, fig_rows * 6))
        axes = axes.flatten()

        for i, col in enumerate(plot_features):
            sns.histplot(real_defaults_inv_df[col], kde=True, color='blue', label='Real Defaults', ax=axes[i], stat='density', alpha=0.6)
            sns.histplot(synth_defaults_inv_df[col], kde=True, color='red', label='Synthetic Defaults (VAE)', ax=axes[i], stat='density', alpha=0.6)
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].legend()
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Density', fontsize=12)
        
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle('Marginal Feature Distribution Comparison (Real vs. Synthetic Defaults)', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(output_filename, dpi=150)
        plt.show()
    else:
        print("No valid features to plot for distribution comparison.")

    return ks_results_df


def perform_correlation_comparison(real_defaults_inv_df, synth_defaults_inv_df, output_filename='correlation_preservation.png'):
    """
    Compares correlation structures between real and synthetic data using Frobenius error and heatmaps.

    Args:
        real_defaults_inv_df (pd.DataFrame): Inverse-transformed real default samples.
        synth_defaults_inv_df (pd.DataFrame): Inverse-transformed synthetic default samples.
        output_filename (str): Filename to save the correlation heatmaps.

    Returns:
        float: Frobenius error between correlation matrices.
    """
    print("\n--- Correlation Structure Preservation (Frobenius Error & Heatmaps) ---")
    if real_defaults_inv_df.empty or synth_defaults_inv_df.empty or len(real_defaults_inv_df.columns) < 2:
        print("Skipping correlation comparison due to empty dataframes or insufficient features.")
        return np.nan

    real_corr = real_defaults_inv_df.corr()
    synth_corr = synth_defaults_inv_df.corr()

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
    plt.savefig(output_filename, dpi=150)
    plt.show()

    return frobenius_error


def perform_privacy_assessment(X_train_scaled_defaults, synth_defaults_df, X_smote_new_scaled, output_filename='privacy_distance_histogram.png', random_state=42):
    """
    Assesses privacy by calculating nearest-neighbor distances between synthetic and real data.

    Args:
        X_train_scaled_defaults (pd.DataFrame): Scaled real default training samples.
        synth_defaults_df (pd.DataFrame): Scaled VAE-generated synthetic default samples.
        X_smote_new_scaled (pd.DataFrame): Scaled SMOTE-generated samples (only the new ones).
        output_filename (str): Filename to save the distance histogram plot.
        random_state (int): Seed for random operations.

    Returns:
        dict: Dictionary containing privacy metrics (mean/min distances).
    """
    print("\n--- Privacy Assessment (Nearest-Neighbor Distance) ---")
    privacy_metrics = {}
    min_distances_vae = np.array([])
    min_distances_smote = np.array([])

    if X_train_scaled_defaults.empty:
        print("No real default samples available for privacy assessment.")
        return privacy_metrics

    if not synth_defaults_df.empty:
        nn_vae = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_vae.fit(X_train_scaled_defaults)
        
        sample_size_vae = min(len(synth_defaults_df), 10000)
        vae_sample = synth_defaults_df.sample(sample_size_vae, random_state=random_state) if sample_size_vae < len(synth_defaults_df) else synth_defaults_df
        distances_vae, _ = nn_vae.kneighbors(vae_sample)
        min_distances_vae = distances_vae.ravel()

        print("Privacy Assessment (VAE synthetic defaults):")
        privacy_metrics['vae_mean_distance'] = np.mean(min_distances_vae)
        privacy_metrics['vae_min_distance'] = np.min(min_distances_vae)
        privacy_metrics['vae_close_records'] = (min_distances_vae < 0.5).sum()
        print(f"  Mean distance to nearest real (VAE): {privacy_metrics['vae_mean_distance']:.4f}")
        print(f"  Min distance (worst case VAE): {privacy_metrics['vae_min_distance']:.4f}")
        print(f"  Records with distance < 0.5 (VAE): {privacy_metrics['vae_close_records']} / {len(min_distances_vae)}")
    else:
        print("VAE synthetic defaults dataframe is empty, skipping VAE privacy assessment.")

    if not X_smote_new_scaled.empty:
        nn_smote = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_smote.fit(X_train_scaled_defaults)
        
        sample_size_smote = min(len(X_smote_new_scaled), 10000)
        smote_sample = X_smote_new_scaled.sample(sample_size_smote, random_state=random_state) if sample_size_smote < len(X_smote_new_scaled) else X_smote_new_scaled
        distances_smote, _ = nn_smote.kneighbors(smote_sample)
        min_distances_smote = distances_smote.ravel()

        print("\nPrivacy Assessment (SMOTE generated records):")
        privacy_metrics['smote_mean_distance'] = np.mean(min_distances_smote)
        privacy_metrics['smote_min_distance'] = np.min(min_distances_smote)
        privacy_metrics['smote_close_records'] = (min_distances_smote < 0.5).sum()
        print(f"  Mean distance to nearest real (SMOTE): {privacy_metrics['smote_mean_distance']:.4f}")
        print(f"  Min distance (worst case SMOTE): {privacy_metrics['smote_min_distance']:.4f}")
        print(f"  Records with distance < 0.5 (SMOTE): {privacy_metrics['smote_close_records']} / {len(min_distances_smote)}")
        print("  (SMOTE samples are generally closer due to linear interpolation)")
    else:
        print("\nSMOTE-generated records not available for privacy comparison or dataframe is empty.")

    if not synth_defaults_df.empty or not X_smote_new_scaled.empty:
        plt.figure(figsize=(10, 6))
        if len(min_distances_vae) > 0:
            sns.histplot(min_distances_vae, kde=True, color='red', label='VAE Synthetic', stat='density', alpha=0.6)
        if len(min_distances_smote) > 0:
            sns.histplot(min_distances_smote, kde=True, color='blue', label='SMOTE Synthetic', stat='density', alpha=0.6)
        
        plt.title('Nearest-Neighbor Distance to Real Defaults (Privacy Assessment)', fontsize=16)
        plt.xlabel('Distance to Closest Real Record', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150)
        plt.show()

    return privacy_metrics


def perform_tsne_visualization(real_defaults_inv_df, synth_defaults_inv_df, n_viz_samples=2000, output_filename='tsne_real_vs_synthetic.png', random_state=42):
    """
    Performs t-SNE visualization to compare real and synthetic data distributions.

    Args:
        real_defaults_inv_df (pd.DataFrame): Inverse-transformed real default samples.
        synth_defaults_inv_df (pd.DataFrame): Inverse-transformed synthetic default samples.
        n_viz_samples (int): Number of samples to use for visualization from each dataset.
        output_filename (str): Filename to save the t-SNE plot.
        random_state (int): Seed for random operations.
    """
    print("\n--- t-SNE / PCA Scatter Plots ---")
    if real_defaults_inv_df.empty or synth_defaults_inv_df.empty:
        print("Skipping t-SNE visualization due to empty dataframes.")
        return

    n_viz_samples = min(len(real_defaults_inv_df), len(synth_defaults_inv_df), n_viz_samples)

    if n_viz_samples == 0:
        print("Not enough samples for t-SNE visualization.")
        return

    real_viz = real_defaults_inv_df.sample(n=n_viz_samples, random_state=random_state)
    synth_viz = synth_defaults_inv_df.sample(n=n_viz_samples, random_state=random_state)

    combined_viz_data = pd.concat([real_viz, synth_viz], ignore_index=True)
    combined_viz_labels = ['Real'] * n_viz_samples + ['Synthetic (VAE)'] * n_viz_samples

    n_components_pca = min(combined_viz_data.shape[1], 5)
    pca_results = combined_viz_data.values
    if n_components_pca > 0: # Only apply PCA if there are features to reduce
        pca = PCA(n_components=n_components_pca, random_state=random_state)
        pca_results = pca.fit_transform(combined_viz_data)
    else:
        print("Warning: Not enough features for PCA, using raw data for t-SNE (if possible).")

    if pca_results.shape[1] > 1 and n_viz_samples > 1: # t-SNE needs at least 2 dimensions and > 1 sample
        # Perplexity must be less than the number of samples
        perplexity_val = min(30, n_viz_samples - 1)
        if perplexity_val <= 0:
            print(f"Cannot perform t-SNE: perplexity ({perplexity_val}) must be greater than 0.")
            return

        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity_val, n_iter=1000, learning_rate=200)
        tsne_results = tsne.fit_transform(pca_results)
    else:
        print("Warning: Data has 1 or fewer dimensions after PCA, or not enough samples for t-SNE to 2D.")
        return

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
    plt.savefig(output_filename, dpi=150)
    plt.show()


def plot_roc_curves(y_test, y_probs_dict, output_filename='roc_curves_comparison.png'):
    """
    Plots ROC curves for multiple models on a single graph.

    Args:
        y_test (pd.Series): True labels for the test set.
        y_probs_dict (dict): Dictionary where keys are model names and values are probability arrays.
        output_filename (str): Filename to save the ROC plot.
    """
    plt.figure(figsize=(10, 8))
    has_valid_probs = False
    for model_name, y_prob in y_probs_dict.items():
        if y_prob is not None and len(y_prob) == len(y_test):
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
            has_valid_probs = True
        else:
            print(f"Warning: Skipping ROC curve for '{model_name}' due to invalid probabilities.")

    if has_valid_probs:
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150)
        plt.show()
    else:
        print("No valid probability predictions to plot ROC curves.")


def generate_comparison_report_and_plots(results_dict, y_test, y_prob_base, y_prob_smote, y_prob_vae, y_prob_tstr=None, output_filename_barchart='augmentation_comparison_barchart.png', output_filename_roc='roc_curves_comparison.png'):
    """
    Generates a consolidated comparison report and plots for model performance.

    Args:
        results_dict (dict): Dictionary containing AUC, AP, TSTR ratio.
        y_test (pd.Series): True labels for the test set.
        y_prob_base (np.array): Probabilities from the baseline model.
        y_prob_smote (np.array): Probabilities from the SMOTE augmented model.
        y_prob_vae (np.array): Probabilities from the VAE augmented model.
        y_prob_tstr (np.array, optional): Probabilities from the TSTR model. Defaults to None.
        output_filename_barchart (str): Filename to save the bar chart.
        output_filename_roc (str): Filename to save the ROC plot.
    """
    comparison_data = pd.DataFrame({
        'Method': ['No Augmentation (Baseline)', 'SMOTE Augmentation', 'VAE Augmentation'],
        'AUC': [results_dict.get('auc_base', np.nan), results_dict.get('auc_smote', np.nan), results_dict.get('auc_vae', np.nan)],
        'Average Precision': [results_dict.get('ap_base', np.nan), results_dict.get('ap_smote', np.nan), results_dict.get('ap_vae', np.nan)],
        'Privacy': ['N/A (real only)', 'Low (Interpolation Risk)', 'Moderate (Distribution Sampling)'],
        'Diversity': ['N/A', 'Low (Convex Hull)', 'High (Learned Distribution)']
    })

    print("\n--- FOUR-WAY AUGMENTATION COMPARISON ---")
    print("=" * 70)
    print(comparison_data.to_string(index=False))
    print("=" * 70)

    if 'tstr_ratio' in results_dict and not np.isnan(results_dict['tstr_ratio']):
        print(f"\nTSTR Ratio (Model trained on VAE synthetic, tested on real): {results_dict['tstr_ratio']:.2%}")

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
    plt.savefig(output_filename_barchart, dpi=150)
    plt.show()

    y_probabilities_dict = {
        'Baseline': y_prob_base,
        'SMOTE Augmented': y_prob_smote,
        'VAE Augmented': y_prob_vae
    }
    if y_prob_tstr is not None and not np.isnan(results_dict.get('auc_tstr', np.nan)):
        y_probabilities_dict['TSTR (Synth-Only)'] = y_prob_tstr

    plot_roc_curves(y_test, y_probabilities_dict, output_filename_roc)


def run_full_synthetic_data_pipeline(
    filepath=None,
    test_size=0.2,
    random_state=42,
    n_samples=100000,
    latent_dim_cvae=8,
    max_synthetic_ratio=5,
    vae_epochs=100,
    vae_batch_size=64,
    vae_validation_split=0.15
):
    """
    Executes the full pipeline for credit default modeling with synthetic data augmentation.

    Args:
        filepath (str, optional): Path to the CSV dataset. If None, synthetic data is generated.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random operations.
        n_samples (int): Number of synthetic samples to generate if filepath is None.
        latent_dim_cvae (int): Dimensionality of the CVAE latent space.
        max_synthetic_ratio (int): Max ratio of synthetic to real defaults for VAE augmentation.
        vae_epochs (int): Number of epochs for CVAE training.
        vae_batch_size (int): Batch size for CVAE training.
        vae_validation_split (float): Validation split for CVAE training.

    Returns:
        dict: A dictionary containing key results (AUC, AP, TSTR ratio, etc.)
    """
    X_train_scaled, X_test_scaled, y_train, y_test, defaults_train, non_defaults_train, scaler = \
        load_and_preprocess_data(filepath, test_size, random_state, n_samples)

    results = {}
    y_probabilities = {}

    # 2. Baseline XGBoost Model (No Augmentation)
    model_base, y_prob_base, auc_base, ap_base = train_and_evaluate_xgboost(
        X_train_scaled, y_train, X_test_scaled, y_test, "Baseline (No Augmentation)"
    )
    results['auc_base'] = auc_base
    results['ap_base'] = ap_base
    y_probabilities['y_prob_base'] = y_prob_base

    # 3. SMOTE Augmentation and XGBoost
    model_smote, y_prob_smote, auc_smote, ap_smote, X_smote_resampled, y_smote_resampled = \
        apply_smote_and_train_xgoost(X_train_scaled, y_train, X_test_scaled, y_test, random_state)
    results['auc_smote'] = auc_smote
    results['ap_smote'] = ap_smote
    y_probabilities['y_prob_smote'] = y_prob_smote

    X_smote_new_scaled = X_smote_resampled.iloc[len(X_train_scaled):] if len(X_smote_resampled) > len(X_train_scaled) else pd.DataFrame(columns=X_train_scaled.columns)

    # 4. Train CVAE
    cvae_model_instance, encoder, decoder, trained_latent_dim = \
        train_cvae(X_train_scaled, y_train, latent_dim=latent_dim_cvae, epochs=vae_epochs, batch_size=vae_batch_size, validation_split=vae_validation_split)

    # 5. VAE Augmentation and XGBoost
    model_vae, y_prob_vae, auc_vae, ap_vae, X_vae_augmented, y_vae_augmented, synth_defaults_df_scaled = \
        augment_with_vae_and_train_xgboost(
            X_train_scaled, y_train, X_test_scaled, y_test,
            defaults_train, non_defaults_train, decoder, trained_latent_dim,
            max_synthetic_ratio
        )
    results['auc_vae'] = auc_vae if auc_vae is not None else np.nan
    results['ap_vae'] = ap_vae if ap_vae is not None else np.nan
    y_probabilities['y_prob_vae'] = y_prob_vae

    synth_defaults_inv_df = pd.DataFrame()
    real_defaults_inv_df = pd.DataFrame()

    if not synth_defaults_df_scaled.empty:
        synth_defaults_raw = scaler.inverse_transform(synth_defaults_df_scaled)
        synth_defaults_inv_df = pd.DataFrame(synth_defaults_raw, columns=X_train_scaled.columns)
    
    if not defaults_train.empty:
        real_defaults_raw = scaler.inverse_transform(defaults_train)
        real_defaults_inv_df = pd.DataFrame(real_defaults_raw, columns=X_train_scaled.columns)

    # 6. TSTR Protocol
    auc_tstr, y_prob_tstr, tstr_ratio = run_tstr_protocol(
        X_train_scaled, y_train, X_test_scaled, y_test, decoder, trained_latent_dim, auc_base, random_state=random_state
    )
    results['auc_tstr'] = auc_tstr
    results['tstr_ratio'] = tstr_ratio
    y_probabilities['y_prob_tstr'] = y_prob_tstr

    # 7. Synthetic Data Quality & Privacy Assessment
    key_features_for_overlay = ['utilization', 'income', 'debt_ratio', 'past_due_30']
    
    ks_results = perform_distribution_comparison(real_defaults_inv_df, synth_defaults_inv_df, key_features_for_overlay)
    results['ks_results'] = ks_results

    frobenius_error = perform_correlation_comparison(real_defaults_inv_df, synth_defaults_inv_df)
    results['frobenius_error'] = frobenius_error

    privacy_metrics = perform_privacy_assessment(
        defaults_train,
        synth_defaults_df_scaled,
        X_smote_new_scaled,
        random_state=random_state
    )
    results['privacy_metrics'] = privacy_metrics

    perform_tsne_visualization(real_defaults_inv_df, synth_defaults_inv_df, random_state=random_state)

    # 8. Consolidated Comparison Report and Plots
    generate_comparison_report_and_plots(
        results, y_test,
        y_probabilities['y_prob_base'],
        y_probabilities['y_prob_smote'],
        y_probabilities['y_prob_vae'],
        y_probabilities['y_prob_tstr']
    )

    return results

if __name__ == "__main__":
    print("Starting the full synthetic data pipeline...")
    pipeline_results = run_full_synthetic_data_pipeline(
        n_samples=50000,
        vae_epochs=50
    )
    print("\nPipeline execution complete. Summary results:")
    print(pipeline_results)
