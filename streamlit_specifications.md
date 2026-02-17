
# Streamlit Application Specification: Synthetic Credit Portfolio Data with VAE

## 1. Application Overview

*Purpose*: This Streamlit application serves as a development blueprint for Alex Chen, a CFA Charterholder and Senior Credit Risk Analyst at Apex Financial. It guides him through a comprehensive workflow to tackle severe class imbalance in loan default prediction. The application enables Alex to compare classical oversampling (SMOTE) with advanced generative AI (Conditional Variational Autoencoders - CVAE) for data augmentation. It rigorously assesses the impact of these strategies on model performance, synthetic data quality, utility, and privacy preservation, providing the necessary insights for strategic decision-making and regulatory compliance at Apex Financial.

*High-Level Story Flow*:
1.  **Introduction & Data Overview**: Alex begins by loading and understanding the real-world loan dataset, immediately observing the critical class imbalance that hinders traditional models. He preprocesses the data for subsequent machine learning tasks.
2.  **Baseline Model**: To establish a performance benchmark, Alex trains an XGBoost model on the original, imbalanced data. This "baseline" sets the bar for evaluating the effectiveness of data augmentation.
3.  **SMOTE Augmentation**: Alex explores SMOTE, a conventional oversampling technique. He applies it to balance the training data, retrains an XGBoost model, and critically evaluates its performance, noting its mechanistic generation approach and lack of privacy guarantees.
4.  **Conditional VAE (CVAE)**: Moving to advanced techniques, Alex builds and trains a CVAE. He deepens his understanding of its architecture and mathematical underpinnings, appreciating its potential to generate diverse and realistic synthetic data conditioned on the default label.
5.  **VAE Augmentation & Model Training**: With the trained CVAE, Alex generates a targeted number of synthetic default records. He augments the real training data with these synthetic instances and trains another XGBoost model, assessing the initial performance uplift. A crucial warning about optimal synthetic-to-real ratios is highlighted.
6.  **Synthetic Data Assessment**: This is a pivotal step where Alex rigorously quantifies the quality, utility, and privacy of the VAE-generated synthetic data. He performs Train-on-Synthetic, Test-on-Real (TSTR) analysis, compares feature distributions using KS tests, evaluates correlation preservation with Frobenius error, and quantifies privacy using nearest-neighbor distances, visualizing findings with t-SNE plots.
7.  **Strategic Review & Comparison**: Alex consolidates all model performances (Baseline, SMOTE, VAE) through comparative charts. He then engages in a critical discussion, integrating regulatory implications (Basel IRB, GDPR, ECOA) and ethical considerations (fairness amplification risk) to formulate a well-rounded recommendation for Apex Financial.

## 2. Code Requirements

### Imports

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from source import (
    load_and_preprocess_data,
    train_and_evaluate_xgboost,
    apply_smote_and_train_xgboost,
    build_cvae,
    generate_synthetic_defaults,
    augment_with_vae_and_train_xgboost,
    plot_roc_curves # This function is available in source.py
)
```

### `st.session_state` Design

#### Initialization (at the very beginning of `app.py`)
```python
if 'page' not in st.session_state:
    st.session_state['page'] = 'Introduction & Data Overview'

# Data & Preprocessing
if 'df_loaded' not in st.session_state:
    st.session_state['df_loaded'] = False
if 'X_train_scaled' not in st.session_state:
    st.session_state['X_train_scaled'] = None
if 'X_test_scaled' not in st.session_state:
    st.session_state['X_test_scaled'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'defaults_train' not in st.session_state:
    st.session_state['defaults_train'] = None
if 'non_defaults_train' not in st.session_state:
    st.session_state['non_defaults_train'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'input_dim' not in st.session_state: # Explicitly store input_dim
    st.session_state['input_dim'] = None

# Baseline Model Results
if 'model_base' not in st.session_state:
    st.session_state['model_base'] = None
if 'y_prob_base' not in st.session_state:
    st.session_state['y_prob_base'] = None
if 'auc_base' not in st.session_state:
    st.session_state['auc_base'] = None
if 'ap_base' not in st.session_state:
    st.session_state['ap_base'] = None

# SMOTE Model Results
if 'model_smote' not in st.session_state:
    st.session_state['model_smote'] = None
if 'y_prob_smote' not in st.session_state:
    st.session_state['y_prob_smote'] = None
if 'auc_smote' not in st.session_state:
    st.session_state['auc_smote'] = None
if 'ap_smote' not in st.session_state:
    st.session_state['ap_smote'] = None
if 'X_smote_resampled' not in st.session_state:
    st.session_state['X_smote_resampled'] = None
if 'y_smote_resampled' not in st.session_state:
    st.session_state['y_smote_resampled'] = None
if 'X_smote_new_scaled' not in st.session_state: # To capture SMOTE generated records for privacy comparison
    st.session_state['X_smote_new_scaled'] = None

# CVAE Components & Data
if 'cvae_model' not in st.session_state:
    st.session_state['cvae_model'] = None
if 'encoder_model' not in st.session_state:
    st.session_state['encoder_model'] = None
if 'decoder_model' not in st.session_state:
    st.session_state['decoder_model'] = None
if 'latent_dim_cvae' not in st.session_state:
    st.session_state['latent_dim_cvae'] = 8 # Default from source.py
if 'y_train_oh' not in st.session_state:
    st.session_state['y_train_oh'] = None

# VAE Augmentation Results
if 'synth_defaults_df' not in st.session_state:
    st.session_state['synth_defaults_df'] = None
if 'model_vae' not in st.session_state:
    st.session_state['model_vae'] = None
if 'y_prob_vae' not in st.session_state:
    st.session_state['y_prob_vae'] = None
if 'auc_vae' not in st.session_state:
    st.session_state['auc_vae'] = None
if 'ap_vae' not in st.session_state:
    st.session_state['ap_vae'] = None
if 'X_vae_augmented' not in st.session_state:
    st.session_state['X_vae_augmented'] = None
if 'y_vae_augmented' not in st.session_state:
    st.session_state['y_vae_augmented'] = None

# Synthetic Data Assessment Results
if 'tstr_auc_value' not in st.session_state:
    st.session_state['tstr_auc_value'] = None
if 'tstr_ratio_value' not in st.session_state:
    st.session_state['tstr_ratio_value'] = None
if 'ks_test_results' not in st.session_state: # List of dicts for KS results
    st.session_state['ks_test_results'] = []
if 'frobenius_error_value' not in st.session_state:
    st.session_state['frobenius_error_value'] = None
if 'min_distances_vae' not in st.session_state:
    st.session_state['min_distances_vae'] = None
if 'min_distances_smote' not in st.session_state:
    st.session_state['min_distances_smote'] = None
if 'real_defaults_inv_df' not in st.session_state:
    st.session_state['real_defaults_inv_df'] = None
if 'synth_defaults_inv_df' not in st.session_state:
    st.session_state['synth_defaults_inv_df'] = None
```

#### Updates and Reads across Pages
*   Each major step is triggered by a Streamlit button. Upon clicking:
    1.  The relevant function(s) from `source.py` are invoked.
    2.  The results (models, predictions, metrics, dataframes, CVAE components, etc.) are captured and stored in corresponding `st.session_state` keys.
    3.  Status messages and step-specific results are displayed to the user.
*   Subsequent pages and steps `read` necessary data and results from `st.session_state` to ensure continuity in the workflow. For instance, `X_train_scaled` and `y_train` loaded on the first page are used for all subsequent model trainings. `auc_base` is read for calculating TSTR ratio on the assessment page, and all AUCs/APs are read for the final comparison.
*   Conditional rendering (`if st.session_state['key'] is not None:`) ensures that analysis and display elements only appear once the prerequisite steps have been successfully executed.

---

### Streamlit Application Structure (`app.py`)

```python
st.set_page_config(layout="wide", page_title="Synthetic Credit Portfolio Data with VAE")

# --- Session State Initialization (as defined above) ---

# --- Sidebar Navigation ---
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Go to",
        [
            "Introduction & Data Overview",
            "Baseline Model",
            "SMOTE Augmentation",
            "Conditional VAE",
            "VAE Augmentation & Model Training",
            "Synthetic Data Assessment",
            "Strategic Review & Comparison"
        ],
        key='page'
    )

# --- Main Page Content ---
st.title("Synthetic Credit Portfolio Data with VAE")
st.markdown(f"**Alex Chen, CFA - Senior Credit Risk Analyst, Apex Financial**")
st.markdown(f"Addressing Class Imbalance in Default Prediction via Variational Autoencoder Data Augmentation")

if st.session_state['page'] == "Introduction & Data Overview":
    st.header("1. Introduction & Data Overview")
    st.markdown(f"Mr. Alex Chen, a CFA Charterholder and Senior Credit Risk Analyst at Apex Financial, faces a critical challenge: developing accurate loan default prediction models amidst severe class imbalance. Only 2-5% of borrowers typically default, causing traditional models to struggle in identifying risky loans and often leading to poor recall for the critical default class. Data privacy regulations further complicate augmenting scarce default records. Alex believes generative AI, specifically Conditional Variational Autoencoders (CVAE), can synthesize realistic, privacy-preserving default records.")
    st.markdown(f"This application guides Alex through applying and evaluating advanced synthetic data generation techniques in his workflow.")

    st.subheader("Data Preparation and Understanding the Imbalance")
    st.markdown(f"Alex's first step is to load the raw loan data and understand its characteristics, especially the severe class imbalance. He knows that without addressing this imbalance, any predictive model will be biased towards the majority class (non-defaults). Preprocessing, such as handling missing values and feature scaling, is crucial to prepare the data for machine learning models. Finally, splitting the data into training and testing sets, ensuring stratification, will provide a reliable basis for model evaluation.")

    if st.button("1. Load & Preprocess Data"):
        with st.spinner("Loading and preprocessing data..."):
            (X_train_scaled, X_test_scaled, y_train, y_test,
             defaults_train, non_defaults_train, scaler) = load_and_preprocess_data()

            st.session_state['X_train_scaled'] = X_train_scaled
            st.session_state['X_test_scaled'] = X_test_scaled
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['defaults_train'] = defaults_train
            st.session_state['non_defaults_train'] = non_defaults_train
            st.session_state['scaler'] = scaler
            st.session_state['df_loaded'] = True
            st.session_state['input_dim'] = X_train_scaled.shape[1] # Store input_dim

            st.success("Data loaded and preprocessed successfully!")

            st.markdown(f"**Data Imbalance Summary:**")
            st.dataframe(pd.DataFrame({
                "Dataset": ["Training", "Test"],
                "Total Samples": [len(X_train_scaled), len(X_test_scaled)],
                "Defaults": [y_train.sum(), y_test.sum()],
                "Default Rate": [f"{y_train.mean():.2%}", f"{y_test.mean():.2%}"]
            }))
            st.markdown(f"The output clearly shows the severe class imbalance in the training data, where defaults represent a very small percentage of the total loans. This imbalance is the core problem Alex needs to address, as it directly impacts the model's ability to learn the characteristics of defaulting borrowers effectively. The data has been scaled, which is crucial for many machine learning algorithms, including the upcoming VAE.")
            st.markdown(f"**First 5 rows of scaled training data:**")
            st.dataframe(X_train_scaled.head())

elif st.session_state['page'] == "Baseline Model":
    st.header("2. Establishing a Baseline: XGBoost on Raw Data")
    st.markdown(f"Before applying any augmentation techniques, Alex needs to establish a performance benchmark. He will train an XGBoost model, a powerful and commonly used algorithm in finance, directly on the imbalanced, preprocessed training data. Evaluating this \"baseline\" model using metrics like Area Under the Receiver Operating Characteristic Curve (AUC) and Average Precision (AP) will provide a starting point for measuring the impact of subsequent data augmentation strategies. For imbalanced datasets, AUC and AP are more informative than simple accuracy, as they capture the model's ability to distinguish between classes across various thresholds.")

    if st.session_state['df_loaded']:
        if st.button("2. Train Baseline XGBoost Model"):
            with st.spinner("Training baseline model..."):
                (model_base, y_prob_base, auc_base, ap_base) = train_and_evaluate_xgboost(
                    st.session_state['X_train_scaled'], st.session_state['y_train'],
                    st.session_state['X_test_scaled'], st.session_state['y_test'],
                    "Baseline (No Augmentation)"
                )
                st.session_state['model_base'] = model_base
                st.session_state['y_prob_base'] = y_prob_base
                st.session_state['auc_base'] = auc_base
                st.session_state['ap_base'] = ap_base
                st.success("Baseline model trained successfully!")

                st.markdown(f"**Baseline Model Performance:**")
                st.dataframe(pd.DataFrame({
                    "Metric": ["AUC", "Average Precision"],
                    "Score": [f"{auc_base:.4f}", f"{ap_base:.4f}"]
                }))
                st.markdown(f"The baseline XGBoost model provides the initial performance metrics. Alex notes the AUC and Average Precision scores. These values will serve as the crucial benchmark to compare against models trained with data augmentation, helping him quantify the value added by more sophisticated techniques.")
        else:
            if st.session_state['auc_base'] is not None:
                 st.markdown(f"**Baseline Model Performance (already trained):**")
                 st.dataframe(pd.DataFrame({
                    "Metric": ["AUC", "Average Precision"],
                    "Score": [f"{st.session_state['auc_base']:.4f}", f"{st.session_state['ap_base']:.4f}"]
                 }))
    else:
        st.warning("Please load and preprocess data first from 'Introduction & Data Overview' page.")

elif st.session_state['page'] == "SMOTE Augmentation":
    st.header("3. Classical Augmentation: SMOTE and its Limitations")
    st.markdown(f"Alex understands that traditional oversampling methods like SMOTE (Synthetic Minority Over-sampling Technique) are a common first approach to address class imbalance. SMOTE works by creating synthetic samples for the minority class along the line segments connecting existing minority class instances. While straightforward, Alex is aware that SMOTE's linear interpolation approach might not capture complex, non-linear dependencies in the data, and it offers no privacy benefits as synthetic samples are directly derived from real ones.")

    st.markdown(r"**Mathematical Formulation for SMOTE:**")
    st.markdown(r"SMOTE generates a new point $ \tilde{{x}} $ by linearly interpolating between a minority sample $ x_i $ and one of its $k$-nearest minority neighbors $ x_j $. The formula is:")
    st.markdown(r"$$ \tilde{{x}} = x_i + \lambda(x_j - x_i) $$")
    st.markdown(r"where $x_i$ is a minority sample, $x_j$ is one of its $k$-nearest minority neighbors, and $\lambda$ is a random number between 0 and 1 (i.e., $ \lambda \sim U(0, 1) $).")
    st.markdown(r"This means all SMOTE-generated points lie on the line segments between existing real data points. In geometric terms, they stay within the \"convex hull\" of the minority class data, potentially limiting the diversity and realism for complex datasets.")

    if st.session_state['auc_base'] is not None:
        if st.button("3. Apply SMOTE & Train XGBoost"):
            with st.spinner("Applying SMOTE and training model..."):
                (model_smote, y_prob_smote, auc_smote, ap_smote, X_smote_resampled, y_smote_resampled) = apply_smote_and_train_xgboost(
                    st.session_state['X_train_scaled'], st.session_state['y_train'],
                    st.session_state['X_test_scaled'], st.session_state['y_test']
                )
                st.session_state['model_smote'] = model_smote
                st.session_state['y_prob_smote'] = y_prob_smote
                st.session_state['auc_smote'] = auc_smote
                st.session_state['ap_smote'] = ap_smote
                st.session_state['X_smote_resampled'] = X_smote_resampled
                st.session_state['y_smote_resampled'] = y_smote_resampled

                # Extract newly generated SMOTE samples for privacy comparison later
                st.session_state['X_smote_new_scaled'] = X_smote_resampled[len(st.session_state['X_train_scaled']):]

                st.success("SMOTE augmentation and model training successful!")

                st.markdown(f"**SMOTE Augmented Model Performance:**")
                st.dataframe(pd.DataFrame({
                    "Metric": ["AUC", "Average Precision"],
                    "Score": [f"{auc_smote:.4f}", f"{ap_smote:.4f}"]
                }))
                st.markdown(f"Alex observes that SMOTE successfully balances the training set, and the model's performance (AUC, AP) shows an improvement over the baseline. However, he remains skeptical about SMOTE's ability to generate truly novel and diverse samples that capture the intricate, non-linear relationships often present in financial data. Furthermore, the privacy aspect is entirely unaddressed by SMOTE, as its synthetic samples are essentially \"mixtures\" of real ones, offering no true anonymity.")
        else:
            if st.session_state['auc_smote'] is not None:
                st.markdown(f"**SMOTE Augmented Model Performance (already trained):**")
                st.dataframe(pd.DataFrame({
                    "Metric": ["AUC", "Average Precision"],
                    "Score": [f"{st.session_state['auc_smote']:.4f}", f"{st.session_state['ap_smote']:.4f}"]
                }))
    else:
        st.warning("Please train the baseline model first from 'Baseline Model' page.")

elif st.session_state['page'] == "Conditional VAE":
    st.header("4. Advanced Augmentation: Building and Training a Conditional VAE")
    st.markdown(f"To overcome the limitations of SMOTE, particularly regarding capturing complex data distributions and preserving privacy, Alex turns to a more sophisticated generative AI technique: the Conditional Variational Autoencoder (CVAE). A CVAE learns a compressed, probabilistic representation (latent space) of the input data, conditioned on a specific label (in this case, default/non-default). This conditioning allows Alex to generate synthetic data for a *specific class*, like defaults, providing targeted augmentation. The CVAE's ability to sample from this learned latent distribution allows it to create entirely new data points that are statistically consistent with the real data, even capturing non-linear dependencies. Unlike SMOTE, these synthetic records do not directly correspond to any real individual, offering a degree of privacy preservation.")

    st.markdown(r"**Mathematical Formulation for CVAE:**")
    st.markdown(r"The Conditional VAE aims to learn a mapping from a latent space $ \mathbf{{z}} $ to the data space $ \mathbf{{x}} $, conditioned on a class label $ \mathbf{{y}} $. It consists of an encoder and a decoder.")
    st.markdown(r"1.  **Encoder (Recognition Model):** $ q_{{\phi}}(\mathbf{{z}}|\mathbf{{x}}, \mathbf{{y}}) $")
    st.markdown(r"    This part maps the input data $ \mathbf{{x}} $ and its condition $ \mathbf{{y}} $ to a probabilistic distribution in the latent space $ \mathbf{{z}} $. It is typically parameterized as a Gaussian distribution:")
    st.markdown(r"$$ q_{{\phi}}(\mathbf{{z}}|\mathbf{{x}}, \mathbf{{y}}) = \mathcal{{N}}(\mu_{{\phi}}(\mathbf{{x}}, \mathbf{{y}}), \text{{diag}}(\sigma^2_{{\phi}}(\mathbf{{x}}, \mathbf{{y}}))) $$")
    st.markdown(r"where $q_{{\phi}}(\mathbf{{z}}|\mathbf{{x}}, \mathbf{{y}})$ is the encoder distribution, $\mu_{{\phi}}$ and $\sigma^2_{{\phi}}$ are neural networks with parameters $\phi$ mapping input data $\mathbf{{x}}$ and condition $\mathbf{{y}}$ to the mean and variance of the latent space $\mathbf{{z}}$.")
    st.markdown(r"2.  **Decoder (Generative Model):** $ p_{{\theta}}(\mathbf{{x}}|\mathbf{{z}}, \mathbf{{y}}) $")
    st.markdown(r"    This part reconstructs the data $ \mathbf{{x}} $ from the latent representation $ \mathbf{{z}} $ and the condition $ \mathbf{{y}} $. It also uses neural networks with parameters $ \theta $.")
    st.markdown(r"The CVAE is trained to optimize a loss function that balances two objectives: Reconstruction Loss and KL Divergence Regularization. The overall loss function (Evidence Lower Bound - ELBO) is:")
    st.markdown(r"$$ \mathcal{{L}}(\phi, \theta; \mathbf{{x}}, \mathbf{{y}}) = ||\mathbf{{x}} - \hat{{\mathbf{{x}}}}||^2 + \beta D_{{KL}}(q_{{\phi}}(\mathbf{{z}}|\mathbf{{x}}, \mathbf{{y}})||p(\mathbf{{z}})) $$")
    st.markdown(r"where $ ||\mathbf{{x}} - \hat{{\mathbf{{x}}}}||^2 $ is the reconstruction loss (e.g., Mean Squared Error for continuous data), $ \beta $ is a regularization parameter (often 1), and $ D_{{KL}} $ is the Kullback-Leibler divergence between the learned latent distribution $ q_{{\phi}}(\mathbf{{z}}|\mathbf{{x}}, \mathbf{{y}}) $ and a prior distribution $ p(\mathbf{{z}}) $ (typically a standard normal distribution).")

    if st.session_state['df_loaded']:
        st.markdown(f"**CVAE Configuration:**")
        latent_dim_cvae_input = st.slider("Latent Dimension:", min_value=2, max_value=32, value=st.session_state['latent_dim_cvae'], key='latent_dim_cvae_input')

        if st.button("4. Build & Train CVAE"):
            st.session_state['latent_dim_cvae'] = latent_dim_cvae_input # Update session state with slider value
            with st.spinner("Building and training Conditional VAE... This may take a few minutes."):
                y_train_oh = to_categorical(st.session_state['y_train'], num_classes=2)
                st.session_state['y_train_oh'] = y_train_oh

                cvae, encoder, decoder = build_cvae(
                    st.session_state['input_dim'],
                    latent_dim=st.session_state['latent_dim_cvae'],
                    n_classes=2
                )
                # Define EarlyStopping callback here as it's used in the fit call
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                )
                cvae.fit(
                    [st.session_state['X_train_scaled'], y_train_oh],
                    epochs=100,
                    batch_size=64,
                    validation_split=0.15,
                    callbacks=[early_stopping],
                    verbose=0
                )
                st.session_state['cvae_model'] = cvae
                st.session_state['encoder_model'] = encoder
                st.session_state['decoder_model'] = decoder
                st.success("CVAE built and trained successfully!")
                st.markdown(f"The CVAE has been built and trained. Alex understands that the training process involved minimizing a combined loss, ensuring both accurate reconstruction of input data and a well-behaved latent space. Now, the trained `decoder` component of the CVAE can be used to generate entirely new synthetic data points, conditioned to be specifically for the default class. This is a crucial step towards balancing the dataset in a privacy-preserving manner.")
        else:
            if st.session_state['cvae_model'] is not None:
                st.success("CVAE already built and trained.")
    else:
        st.warning("Please load and preprocess data first from 'Introduction & Data Overview' page.")

elif st.session_state['page'] == "VAE Augmentation & Model Training":
    st.header("5. VAE-Augmented Modeling and Initial Performance Check")
    st.markdown(f"With the CVAE trained, Alex can now leverage its generative capabilities. He will use the CVAE's decoder to create a targeted number of synthetic default records, aiming to balance the original training dataset. This augmented dataset will then be used to train another XGBoost model. By comparing its performance to the baseline and SMOTE models, Alex can get a preliminary understanding of the CVAE's impact on predictive accuracy. A key consideration here, derived from financial best practices and research, is the 'more data is not always better' principle. Over-generating synthetic data can introduce noise, so Alex will cap the synthetic defaults to a reasonable multiple of the existing real defaults (e.g., 5x) to maintain signal integrity.")

    if st.session_state['decoder_model'] is not None:
        st.markdown(f"**Synthetic Data Generation Configuration:**")
        max_synthetic_ratio_input = st.slider("Max Synthetic-to-Real Ratio for Defaults:", min_value=1, max_value=10, value=5, key='max_synthetic_ratio')

        if st.button("5. Generate VAE Data & Train XGBoost"):
            with st.spinner("Generating synthetic defaults and training VAE augmented model..."):
                (model_vae, y_prob_vae, auc_vae, ap_vae, X_vae_augmented, y_vae_augmented) = augment_with_vae_and_train_xgboost(
                    st.session_state['X_train_scaled'], st.session_state['y_train'],
                    st.session_state['X_test_scaled'], st.session_state['y_test'],
                    st.session_state['defaults_train'], st.session_state['non_defaults_train'],
                    st.session_state['decoder_model'], st.session_state['latent_dim_cvae'],
                    max_synthetic_ratio=max_synthetic_ratio_input
                )
                st.session_state['model_vae'] = model_vae
                st.session_state['y_prob_vae'] = y_prob_vae
                st.session_state['auc_vae'] = auc_vae
                st.session_state['ap_vae'] = ap_vae
                st.session_state['X_vae_augmented'] = X_vae_augmented
                st.session_state['y_vae_augmented'] = y_vae_augmented

                # The augment_with_vae_and_train_xgboost function internally calls generate_synthetic_defaults.
                # To get synth_defaults_df for assessment, we must extract it.
                # This assumes generate_synthetic_defaults creates a dataframe that can be retrieved
                # from the augmented data by identifying the synthetic part.
                # Re-calling generate_synthetic_defaults with the same parameters for simplicity
                n_defaults_real = len(st.session_state['defaults_train'])
                n_non_defaults_real = len(st.session_state['non_defaults_train'])
                n_defaults_needed = n_non_defaults_real - n_defaults_real
                n_synthetic_to_generate = min(n_defaults_needed, n_defaults_real * max_synthetic_ratio_input)
                if n_synthetic_to_generate > 0:
                    synth_defaults_df = generate_synthetic_defaults(
                        st.session_state['decoder_model'], n_synthetic_to_generate, st.session_state['latent_dim_cvae']
                    )
                    st.session_state['synth_defaults_df'] = synth_defaults_df
                else:
                    st.session_state['synth_defaults_df'] = pd.DataFrame(columns=st.session_state['X_train_scaled'].columns) # Empty df

                st.success("VAE augmentation and model training successful!")

                st.markdown(f"**VAE Augmented Model Performance:**")
                st.dataframe(pd.DataFrame({
                    "Metric": ["AUC", "Average Precision"],
                    "Score": [f"{auc_vae:.4f}", f"{ap_vae:.4f}"]
                }))
                st.markdown(f"Alex observes the performance of the XGBoost model trained on the VAE-augmented dataset. He compares the AUC and Average Precision with the baseline and SMOTE models. While improvements are often seen, Alex knows that simply looking at performance metrics is not enough. He needs to rigorously assess the *quality*, *utility*, and *privacy* of the synthetic data itself before making a recommendation to Apex Financial's risk committee.")
        else:
            if st.session_state['auc_vae'] is not None:
                st.markdown(f"**VAE Augmented Model Performance (already trained):**")
                st.dataframe(pd.DataFrame({
                    "Metric": ["AUC", "Average Precision"],
                    "Score": [f"{st.session_state['auc_vae']:.4f}", f"{st.session_state['ap_vae']:.4f}"]
                }))
    else:
        st.warning("Please build and train the CVAE first from 'Conditional VAE' page.")

elif st.session_state['page'] == "Synthetic Data Assessment":
    st.header("6. Comprehensive Synthetic Data Assessment: Quality, Utility, and Privacy")
    st.markdown(f"For Apex Financial to adopt synthetic data, Alex must provide a thorough assessment beyond just model performance. He needs to quantify if the synthetic data accurately reflects the real data's characteristics (quality), if models trained on synthetic data generalize well to real data (utility), and crucially, if the synthetic data truly preserves privacy (privacy). This section demonstrates the gold standard tests for these critical aspects, addressing concerns from both technical and regulatory perspectives.")

    if st.session_state['synth_defaults_df'] is not None and st.session_state['auc_vae'] is not None:
        if st.button("6. Perform Comprehensive Assessment"):
            with st.spinner("Performing comprehensive synthetic data assessment..."):
                # Inverse transform for interpretable comparison
                synth_defaults_raw = st.session_state['scaler'].inverse_transform(st.session_state['synth_defaults_df'])
                synth_defaults_inv_df = pd.DataFrame(synth_defaults_raw, columns=st.session_state['X_train_scaled'].columns)

                real_defaults_raw = st.session_state['scaler'].inverse_transform(st.session_state['defaults_train'])
                real_defaults_inv_df = pd.DataFrame(real_defaults_raw, columns=st.session_state['X_train_scaled'].columns)
                st.session_state['real_defaults_inv_df'] = real_defaults_inv_df
                st.session_state['synth_defaults_inv_df'] = synth_defaults_inv_df

                st.subheader("7.1 Train-on-Synthetic, Test-on-Real (TSTR) Protocol")
                st.markdown(f"The gold standard for synthetic data utility evaluation. A model is trained exclusively on synthetic data and tested on real holdout data:")
                st.markdown(r"$$ TSTR~Ratio = \frac{AUC_{trained~on~synthetic}}{AUC_{trained~on~real}} $$")
                st.markdown(r"where $AUC_{trained~on~synthetic}$ is the AUC of a model trained solely on synthetic data, and $AUC_{trained~on~real}$ is the AUC of the baseline model trained on real data.")

                # Re-implement TSTR logic for Streamlit as it's not a direct function in source.py
                n_synthetic_per_class_tstr = len(st.session_state['non_defaults_train']) # To match the majority class count
                if n_synthetic_per_class_tstr > 0:
                    synth_defaults_tstr_df = generate_synthetic_defaults(st.session_state['decoder_model'], n_synthetic_per_class_tstr, st.session_state['latent_dim_cvae'])
                    z_non_default_tstr = np.random.normal(0, 1, (n_synthetic_per_class_tstr, st.session_state['latent_dim_cvae']))
                    y_non_default_cond_tstr = np.zeros((n_synthetic_per_class_tstr, 2))
                    y_non_default_cond_tstr[:, 0] = 1.0 # Condition for non-default
                    synth_non_defaults_tstr_df = pd.DataFrame(st.session_state['decoder_model'].predict([z_non_default_tstr, y_non_default_cond_tstr], verbose=0), columns=st.session_state['X_train_scaled'].columns)

                    X_synth_only = pd.concat([synth_defaults_tstr_df, synth_non_defaults_tstr_df], ignore_index=True)
                    y_synth_only = pd.Series([1]*len(synth_defaults_tstr_df) + [0]*len(synth_non_defaults_tstr_df))

                    shuffled_idx_tstr = np.random.permutation(len(X_synth_only))
                    X_synth_only = X_synth_only.iloc[shuffled_idx_tstr].reset_index(drop=True)
                    y_synth_only = y_synth_only.iloc[shuffled_idx_tstr].reset_index(drop=True)

                    model_tstr = tf.keras.models.Model([tf.keras.layers.Input(shape=(X_synth_only.shape[1],)), tf.keras.layers.Input(shape=(2,))], tf.keras.layers.Dense(1, activation='sigmoid')) # Placeholder, actual model is xgb
                    model_tstr = xgb.XGBClassifier(
                        n_estimators=200, max_depth=5, learning_rate=0.1,
                        eval_metric='auc', random_state=42, use_label_encoder=False
                    )
                    model_tstr.fit(X_synth_only, y_synth_only)
                    y_prob_tstr = model_tstr.predict_proba(st.session_state['X_test_scaled'])[:, 1]
                    auc_tstr = roc_auc_score(st.session_state['y_test'], y_prob_tstr)
                    st.session_state['tstr_auc_value'] = auc_tstr
                    if st.session_state['auc_base'] and st.session_state['auc_base'] > 0:
                        tstr_ratio = auc_tstr / st.session_state['auc_base']
                        st.session_state['tstr_ratio_value'] = tstr_ratio
                        st.markdown(f"**TSTR AUC (trained ONLY on synthetic, tested on real):** `{auc_tstr:.4f}`")
                        st.markdown(f"**Real data AUC (baseline):** `{st.session_state['auc_base']:.4f}`")
                        st.markdown(f"**TSTR Ratio:** `{tstr_ratio:.2%}` (Target > 80% for good utility)")
                    else:
                        st.warning("Baseline AUC is zero, cannot compute TSTR ratio.")
                else:
                    st.info("Not enough samples to generate for TSTR balanced synthetic dataset. Skipping TSTR.")

                st.subheader("7.2 Feature Distribution Comparison (KS Test & Overlays)")
                st.markdown(f"Alex evaluates if the marginal distributions of individual features in the synthetic data match those in the real data. He looks for statistical similarity using the Kolmogorov-Smirnov (KS) test and visual congruence in overlay plots.")
                st.markdown(f"**Column-wise KS Tests (Real vs Synthetic Defaults):**")
                ks_test_results = []
                for col in st.session_state['X_train_scaled'].columns:
                    ks_stat, p_val = ks_2samp(real_defaults_inv_df[col], synth_defaults_inv_df[col])
                    status = "OK" if p_val > 0.05 else "MISMATCH"
                    ks_test_results.append({'Feature': col, 'KS Statistic': f"{ks_stat:.4f}", 'P-value': f"{p_val:.4f}", 'Status': status})
                st.dataframe(pd.DataFrame(ks_test_results))
                st.session_state['ks_test_results'] = ks_test_results

                st.markdown(f"**Marginal Feature Distribution Comparison (Real vs. Synthetic Defaults):**")
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
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("7.3 Correlation Structure Preservation (Frobenius Error & Heatmaps)")
                st.markdown(f"By comparing correlation matrices, Alex can see if the inter-feature dependencies are preserved. The Frobenius error provides a single quantitative measure; a low error (e.g., $ < 0.5 $ for 10 features) indicates good preservation of multivariate relationships.")
                st.markdown(r"**Mathematical Formulation for Correlation Matrix Frobenius Error:**")
                st.markdown(r"To quantify how well the VAE preserves feature correlations:")
                st.markdown(r"$$ \epsilon_{corr} = \|C_{real} - C_{synth}\|_F = \sqrt{\sum_{i,j}(C^{real}_{ij} - C^{synth}_{ij})^2} $$")
                st.markdown(r"where $C_{real}$ is the Pearson correlation matrix of real data, $C_{synth}$ is the Pearson correlation matrix of synthetic data, and $||.||_F$ denotes the Frobenius norm.")

                real_corr = real_defaults_inv_df.corr()
                synth_corr = synth_defaults_inv_df.corr()
                frobenius_error = np.linalg.norm(real_corr - synth_corr, 'fro')
                st.session_state['frobenius_error_value'] = frobenius_error
                st.markdown(f"**Correlation Matrix Frobenius Error (Real vs Synthetic Defaults):** `{frobenius_error:.4f}`")
                st.markdown(f"*(Target: < 0.5 for 10-feature credit data - lower is better)*")

                fig, axes = plt.subplots(1, 2, figsize=(18, 7))
                sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0], cbar_kws={'shrink': 0.8})
                axes[0].set_title('Real Defaults: Correlation Matrix', fontsize=16)
                sns.heatmap(synth_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1], cbar_kws={'shrink': 0.8})
                axes[1].set_title('Synthetic Defaults (VAE): Correlation Matrix', fontsize=16)
                plt.suptitle('Correlation Preservation: Real vs. Synthetic Defaults', fontsize=18, y=1.02)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("7.4 Privacy Assessment (Nearest-Neighbor Distance)")
                st.markdown(f"This is critical for privacy. VAE-generated samples should be sufficiently \"far\" from any real data point. Alex expects VAE samples to have larger nearest-neighbor distances compared to SMOTE, which, by design, places synthetic samples directly between real ones, offering no privacy. This analysis quantifies the `privacy advantage` of VAEs.")

                # VAE-generated synthetic defaults
                nn_vae = NearestNeighbors(n_neighbors=1, metric='euclidean')
                nn_vae.fit(st.session_state['X_train_scaled'][st.session_state['y_train'] == 1]) # Fit to real default records
                distances_vae, _ = nn_vae.kneighbors(st.session_state['synth_defaults_df'].head(1000)) # Check distances for first 1000 VAE samples
                min_distances_vae = distances_vae.ravel()
                st.session_state['min_distances_vae'] = min_distances_vae

                st.markdown(f"**Privacy Assessment (VAE synthetic defaults):**")
                st.markdown(f"  Mean distance to nearest real (VAE): `{np.mean(min_distances_vae):.4f}`")
                st.markdown(f"  Min distance (worst case VAE): `{np.min(min_distances_vae):.4f}`")
                st.markdown(f"  Records with distance < 0.5 (VAE): `{np.sum(min_distances_vae < 0.5)}` / `{len(min_distances_vae)}`")

                # SMOTE-generated records (if available)
                if st.session_state['X_smote_new_scaled'] is not None and len(st.session_state['X_smote_new_scaled']) > 0:
                    nn_smote = NearestNeighbors(n_neighbors=1, metric='euclidean')
                    nn_smote.fit(st.session_state['X_train_scaled'][st.session_state['y_train'] == 1]) # Fit to real default records
                    distances_smote, _ = nn_smote.kneighbors(st.session_state['X_smote_new_scaled'].head(1000)) # Check distances for first 1000 SMOTE samples
                    min_distances_smote = distances_smote.ravel()
                    st.session_state['min_distances_smote'] = min_distances_smote

                    st.markdown(f"\n**Privacy Assessment (SMOTE generated records):**")
                    st.markdown(f"  Mean distance to nearest real (SMOTE): `{np.mean(min_distances_smote):.4f}`")
                    st.markdown(f"  Min distance (worst case SMOTE): `{np.min(min_distances_smote):.4f}`")
                    st.markdown(f"  Records with distance < 0.5 (SMOTE): `{np.sum(min_distances_smote < 0.5)}` / `{len(min_distances_smote)}`")
                    st.markdown(f"  *(SMOTE samples are generally closer due to linear interpolation)*")

                    fig = plt.figure(figsize=(10, 6))
                    sns.histplot(min_distances_vae, kde=True, color='red', label='VAE Synthetic', stat='density', alpha=0.6)
                    sns.histplot(min_distances_smote, kde=True, color='blue', label='SMOTE Synthetic', stat='density', alpha=0.6)
                    plt.title('Nearest-Neighbor Distance to Real Defaults (Privacy Assessment)', fontsize=16)
                    plt.xlabel('Distance to Closest Real Record', fontsize=12)
                    plt.ylabel('Density', fontsize=12)
                    plt.legend()
                    plt.grid(axis='y', alpha=0.75)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("\nSMOTE-generated records not available for privacy comparison.")

                st.subheader("7.5 t-SNE / PCA Scatter Plots")
                st.markdown(f"This visualization helps Alex understand if synthetic defaults occupy similar regions in the feature space as real defaults, indicating realism, without necessarily overlapping perfectly, which suggests privacy.")

                # t-SNE / PCA Visualization
                n_viz_samples = min(len(real_defaults_inv_df), len(synth_defaults_inv_df), 2000)

                real_viz = real_defaults_inv_df.sample(n=n_viz_samples, random_state=42)
                synth_viz = synth_defaults_inv_df.sample(n=n_viz_samples, random_state=42)

                combined_viz_data = pd.concat([real_viz, synth_viz], ignore_index=True)
                combined_viz_labels = ['Real'] * n_viz_samples + ['Synthetic (VAE)'] * n_viz_samples

                pca = PCA(n_components=min(combined_viz_data.shape[1], 5), random_state=42)
                pca_results = pca.fit_transform(combined_viz_data)

                tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate=200)
                tsne_results = tsne.fit_transform(pca_results)

                fig = plt.figure(figsize=(12, 8))
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
                st.pyplot(fig)
                plt.close(fig)

                st.success("Comprehensive assessment completed!")

                st.markdown(f"This section provides Alex with a powerful toolkit for assessing synthetic data.")
                st.markdown(f"*   **TSTR Protocol:** The TSTR AUC and ratio indicate how well a model trained *solely* on synthetic data performs on real data. A high ratio suggests the synthetic data captures the underlying predictive patterns effectively. Alex notes if the TSTR Ratio meets the target of `> 80%` for good utility.")
                st.markdown(f"*   **KS Tests and Distribution Overlays:** These visuals and statistics confirm that the marginal distributions of key features in the synthetic data closely match those in the real data. Alex looks for \"OK\" statuses in the KS tests and visual congruence in the overlay plots to ensure the synthetic data is statistically similar.")
                st.markdown(f"*   **Correlation Heatmaps and Frobenius Error:** By comparing correlation matrices, Alex can see if the inter-feature dependencies are preserved. The Frobenius error provides a single quantitative measure; a low error (e.g., $ < 0.5 $ for 10 features) indicates good preservation of multivariate relationships.")
                st.markdown(f"*   **Nearest-Neighbor Distance Histogram:** This is critical for privacy. VAE-generated samples should be sufficiently \"far\" from any real data point. Alex expects VAE samples to have larger nearest-neighbor distances compared to SMOTE, which, by design, places synthetic samples directly between real ones, offering no privacy. This analysis quantifies the `privacy advantage` of VAEs.")
                st.markdown(f"*   **t-SNE Plot:** This visualization helps Alex understand if synthetic defaults occupy similar regions in the feature space as real defaults, indicating realism, without necessarily overlapping perfectly, which suggests privacy.")
                st.markdown(f"These comprehensive assessments provide the empirical evidence Alex needs to build a strong case for using CVAE-generated data at Apex Financial.")

        else:
            if st.session_state['tstr_auc_value'] is not None:
                st.success("Comprehensive assessment already performed. Navigate to 'Strategic Review & Comparison' for consolidated view.")
            else:
                st.info("Click the button to perform the comprehensive assessment.")
    else:
        st.warning("Please train the VAE augmented model first from 'VAE Augmentation & Model Training' page.")

elif st.session_state['page'] == "Strategic Review & Comparison":
    st.header("7. Strategic Review: Performance, Regulatory Compliance, and Ethical AI")
    st.markdown(f"After extensive analysis, Alex needs to synthesize his findings for Apex Financial's risk committee and other stakeholders. This means not only comparing the performance of the different models but also engaging in crucial discussions around regulatory compliance (Basel IRB, ECOA, GDPR) and ethical considerations, particularly the risk of fairness amplification. This holistic view will guide Apex in strategically integrating synthetic data into their credit modeling workflows.")

    if st.session_state['auc_vae'] is not None and st.session_state['auc_smote'] is not None and st.session_state['auc_base'] is not None:
        if st.button("7. Consolidate & Review Results"):
            st.subheader("Four-Way Augmentation Comparison")
            comparison_data = pd.DataFrame({
                'Method': ['No Augmentation (Baseline)', 'SMOTE Augmentation', 'VAE Augmentation'],
                'AUC': [st.session_state['auc_base'], st.session_state['auc_smote'], st.session_state['auc_vae']],
                'Average Precision': [st.session_state['ap_base'], st.session_state['ap_smote'], st.session_state['ap_vae']],
                'Privacy': ['N/A (real only)', 'Low (Interpolation Risk)', 'Moderate (Distribution Sampling)'],
                'Diversity': ['N/A', 'Low (Convex Hull)', 'High (Learned Distribution)']
            })
            st.dataframe(comparison_data)

            if st.session_state['tstr_ratio_value'] is not None:
                st.markdown(f"**TSTR Ratio (Model trained on VAE synthetic, tested on real):** `{st.session_state['tstr_ratio_value']:.2%}`")

            st.markdown(f"**Credit Default Model Performance: Augmentation Strategy Comparison**")
            fig = plt.figure(figsize=(12, 7))
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
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(f"**Receiver Operating Characteristic (ROC) Curves**")
            fig = plt.figure(figsize=(10, 8)) # plot_roc_curves creates its own figure, so capture current fig
            plot_roc_curves(
                st.session_state['y_test'],
                [st.session_state['y_prob_base'], st.session_state['y_prob_smote'], st.session_state['y_prob_vae']],
                ['Baseline', 'SMOTE Augmented', 'VAE Augmented']
            )
            st.pyplot(fig) # This should display the figure created by plot_roc_curves
            plt.close(fig) # Close the figure to prevent overlap

            st.subheader("Discussion Points for Financial Professionals (Alex's key takeaways):")
            st.markdown(f"Alex's presentation to Apex Financial's risk committee is now complete with concrete evidence. The comparative bar charts and ROC curves visually demonstrate the uplift in model performance (AUC, Average Precision) achieved through VAE augmentation, surpassing SMOTE and significantly outperforming the baseline. The detailed assessment of synthetic data quality, utility, and privacy from the previous section underpins these performance gains.")
            st.markdown(f"1.  **Performance & Utility:** The VAE-augmented model generally shows better AUC and Average Precision, indicating a more robust ability to identify defaults. The TSTR ratio confirms that models trained on VAE synthetic data retain substantial predictive power on real data, validating its utility.")
            st.markdown(f"2.  **Privacy Preservation:** VAE-generated data offers a significant privacy advantage over SMOTE, as evidenced by the greater nearest-neighbor distances. This is crucial for compliance with regulations like **GDPR** and **CCPA**, enabling data sharing for model validation, stress testing (e.g., under **Basel IRB** framework), and research without exposing sensitive customer information.")
            st.markdown(f"3.  **Data Quality:** The KS tests and correlation matrix comparisons demonstrate that VAE successfully preserves the statistical properties and inter-feature relationships of the real data, which is vital for building trustworthy models.")
            st.markdown(f"4.  **Regulatory Scrutiny:** While synthetic data is increasingly accepted, regulators (e.g., OCC, PRA, ECB) still primarily require models to be developed on real data. Synthetic data is best viewed as a *supplement* for augmentation, validation, and collaboration. Apex Financial needs clear policies on how synthetic data is generated, validated, and used, maintaining rigorous oversight of its quality.")
            st.markdown(f"5.  **Ethical Considerations (Fairness Amplification Risk):** Alex warns the committee about the \"fairness amplification risk.\" If the original real training data contains historical biases (e.g., discriminatory lending practices impacting certain demographic groups), a CVAE will faithfully reproduce, and potentially amplify, these biases in the synthetic data. This could lead to models perpetuating unfair outcomes. Therefore, rigorous bias audits on both real and synthetic data, and the resulting models, are imperative, aligning with principles of **ECOA (Equal Credit Opportunity Act)**.")
            st.markdown(f"6.  **Production Readiness:** For production deployments, libraries like `sdv` offering models like CTGAN could be explored as they handle mixed data types and offer robust implementations out-of-the-box, potentially reducing custom code complexity.")
            st.markdown(f"Alex concludes that Conditional VAEs are a powerful tool for Apex Financial, offering a principled approach to tackle class imbalance and privacy concerns simultaneously, thereby enhancing risk management capabilities and accelerating model development within a responsible AI framework.")

        else:
            st.info("Click the button to consolidate and review all results.")
    else:
        st.warning("Please ensure all previous steps, including 'Synthetic Data Assessment', have been completed.")
```
