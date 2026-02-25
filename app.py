import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from source import (
    load_and_preprocess_data,
    train_and_evaluate_xgboost,
    apply_smote_and_train_xgboost,
    train_cvae,
    generate_synthetic_defaults,
    augment_with_vae_and_train_xgboost,
    run_tstr_protocol,
    perform_distribution_comparison,
    perform_correlation_comparison,
    perform_privacy_assessment,
    perform_tsne_visualization,
    plot_roc_curves,
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    layout="wide", page_title="Synthetic Credit Portfolio Data with VAE")

st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
# -----------------------------
# Session State Initialization
# -----------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Introduction & Data Overview"

# Data & preprocessing
if "df_loaded" not in st.session_state:
    st.session_state["df_loaded"] = False
if "X_train_scaled" not in st.session_state:
    st.session_state["X_train_scaled"] = None
if "X_test_scaled" not in st.session_state:
    st.session_state["X_test_scaled"] = None
if "y_train" not in st.session_state:
    st.session_state["y_train"] = None
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None
if "defaults_train" not in st.session_state:
    st.session_state["defaults_train"] = None
if "non_defaults_train" not in st.session_state:
    st.session_state["non_defaults_train"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "input_dim" not in st.session_state:
    st.session_state["input_dim"] = None

# Baseline model results
if "model_base" not in st.session_state:
    st.session_state["model_base"] = None
if "y_prob_base" not in st.session_state:
    st.session_state["y_prob_base"] = None
if "auc_base" not in st.session_state:
    st.session_state["auc_base"] = None
if "ap_base" not in st.session_state:
    st.session_state["ap_base"] = None

# SMOTE model results
if "model_smote" not in st.session_state:
    st.session_state["model_smote"] = None
if "y_prob_smote" not in st.session_state:
    st.session_state["y_prob_smote"] = None
if "auc_smote" not in st.session_state:
    st.session_state["auc_smote"] = None
if "ap_smote" not in st.session_state:
    st.session_state["ap_smote"] = None
if "X_smote_resampled" not in st.session_state:
    st.session_state["X_smote_resampled"] = None
if "y_smote_resampled" not in st.session_state:
    st.session_state["y_smote_resampled"] = None
if "X_smote_new_scaled" not in st.session_state:
    st.session_state["X_smote_new_scaled"] = None

# CVAE components & data
if "cvae_model" not in st.session_state:
    st.session_state["cvae_model"] = None
if "encoder_model" not in st.session_state:
    st.session_state["encoder_model"] = None
if "decoder_model" not in st.session_state:
    st.session_state["decoder_model"] = None
if "latent_dim_cvae" not in st.session_state:
    st.session_state["latent_dim_cvae"] = 8  # default used in source.py
if "y_train_oh" not in st.session_state:
    st.session_state["y_train_oh"] = None

# VAE augmentation results
if "synth_defaults_df" not in st.session_state:
    st.session_state["synth_defaults_df"] = None
if "model_vae" not in st.session_state:
    st.session_state["model_vae"] = None
if "y_prob_vae" not in st.session_state:
    st.session_state["y_prob_vae"] = None
if "auc_vae" not in st.session_state:
    st.session_state["auc_vae"] = None
if "ap_vae" not in st.session_state:
    st.session_state["ap_vae"] = None
if "X_vae_augmented" not in st.session_state:
    st.session_state["X_vae_augmented"] = None
if "y_vae_augmented" not in st.session_state:
    st.session_state["y_vae_augmented"] = None

# Synthetic data assessment results
if "tstr_auc_value" not in st.session_state:
    st.session_state["tstr_auc_value"] = None
if "tstr_ratio_value" not in st.session_state:
    st.session_state["tstr_ratio_value"] = None
if "ks_test_results" not in st.session_state:
    st.session_state["ks_test_results"] = []
if "frobenius_error_value" not in st.session_state:
    st.session_state["frobenius_error_value"] = None
if "min_distances_vae" not in st.session_state:
    st.session_state["min_distances_vae"] = None
if "min_distances_smote" not in st.session_state:
    st.session_state["min_distances_smote"] = None
if "real_defaults_inv_df" not in st.session_state:
    st.session_state["real_defaults_inv_df"] = None
if "synth_defaults_inv_df" not in st.session_state:
    st.session_state["synth_defaults_inv_df"] = None
if "assessment_ready" not in st.session_state:
    st.session_state["assessment_ready"] = False


# -----------------------------
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    # Optional logo (don’t crash if missing)
    logo_path_candidates = [
        "assets/images/company_logo.jpg",
        "assets/company_logo.jpg",
        "company_logo.jpg",
    ]
    for lp in logo_path_candidates:
        if os.path.exists(lp):
            st.image(lp)
            break

    st.header("Navigation")
    st.session_state["page"] = st.selectbox(
        "Go to",
        [
            "Introduction & Data Overview",
            "Baseline Model",
            "SMOTE Augmentation",
            "Conditional VAE",
            "VAE Augmentation & Model Training",
            "Synthetic Data Assessment",
            "Strategic Review & Comparison",
        ],
    )


# -----------------------------
# App Header
# -----------------------------
st.title("Synthetic Credit Portfolio Data with VAE")
st.markdown("**Alex Chen, CFA – Senior Credit Risk Analyst, Apex Financial**")
st.markdown(
    "Addressing Class Imbalance in Default Prediction via Conditional Variational Autoencoder (CVAE)")
st.divider()


# -----------------------------
# Page 1: Introduction & Data Overview
# -----------------------------
if st.session_state["page"] == "Introduction & Data Overview":
    st.header("1. Introduction & Data Overview")

    st.markdown(
        """
Mr. Alex Chen, a CFA Charterholder and Senior Credit Risk Analyst at **Apex Financial**, faces a familiar but high-stakes problem:
**loan defaults are rare** (often only **2–5%** of borrowers), which causes standard ML models to “learn” the majority class and
miss the cases that matter most.

At the same time, **privacy and compliance** constraints make it difficult to share or expand sensitive default examples.
Alex suspects that **generative AI**—specifically a **Conditional Variational Autoencoder (CVAE)**—could create realistic,
privacy-preserving synthetic defaults to strengthen model training and validation workflows.
"""
    )

    st.subheader("Data Preparation and Understanding the Imbalance")
    st.markdown(
        """
Alex’s first step is to load the loan dataset and prepare it for modeling:

- Handle missing values and ensure consistent feature types.
- Scale numeric features for stability (critical for neural generative models).
- Create stratified train/test splits so evaluation remains realistic.
- Explicitly quantify the default imbalance to frame the modeling challenge.
"""
    )

    if st.button("1. Load & Preprocess Data"):
        with st.spinner("Loading and preprocessing data..."):
            (
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                defaults_train,
                non_defaults_train,
                scaler,
            ) = load_and_preprocess_data()

            st.session_state["X_train_scaled"] = X_train_scaled
            st.session_state["X_test_scaled"] = X_test_scaled
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.session_state["defaults_train"] = defaults_train
            st.session_state["non_defaults_train"] = non_defaults_train
            st.session_state["scaler"] = scaler
            st.session_state["df_loaded"] = True
            st.session_state["input_dim"] = X_train_scaled.shape[1]

        st.success("Data loaded and preprocessed successfully!")

        st.markdown("**Data Imbalance Summary:**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Dataset": ["Training", "Test"],
                    "Total Samples": [len(st.session_state["X_train_scaled"]), len(st.session_state["X_test_scaled"])],
                    "Defaults": [int(st.session_state["y_train"].sum()), int(st.session_state["y_test"].sum())],
                    "Default Rate": [f"{st.session_state['y_train'].mean():.2%}", f"{st.session_state['y_test'].mean():.2%}"],
                }
            )
        )

        st.markdown(
            """
The output above typically shows a **severe class imbalance**—the core issue Alex must solve.
Scaling has been applied, which is essential for stable optimization in both XGBoost pipelines and VAE training.
"""
        )

        st.markdown("**First 5 rows of scaled training data:**")
        st.dataframe(st.session_state["X_train_scaled"].head())

    elif st.session_state["df_loaded"]:
        st.info(
            "Data is already loaded. You can proceed to the next steps using the sidebar.")


# -----------------------------
# Page 2: Baseline Model
# -----------------------------
elif st.session_state["page"] == "Baseline Model":
    st.header("2. Establishing a Baseline: XGBoost on Raw (Imbalanced) Data")

    st.markdown(
        """
Before any augmentation, Alex needs a **benchmark**.

He trains an **XGBoost** model directly on the imbalanced data to measure baseline discrimination. For imbalanced default modeling,
**Accuracy is misleading**, so Alex focuses on:

- **AUC (ROC AUC):** ranking quality across thresholds
- **Average Precision (AP):** precision-recall quality; more informative when positives are rare
"""
    )

    if not st.session_state["df_loaded"]:
        st.warning(
            "Please load and preprocess data first from 'Introduction & Data Overview'.")
    else:
        if st.button("2. Train Baseline XGBoost Model"):
            with st.spinner("Training baseline model..."):
                model_base, y_prob_base, auc_base, ap_base = train_and_evaluate_xgboost(
                    st.session_state["X_train_scaled"],
                    st.session_state["y_train"],
                    st.session_state["X_test_scaled"],
                    st.session_state["y_test"],
                    "Baseline (No Augmentation)",
                )

                st.session_state["model_base"] = model_base
                st.session_state["y_prob_base"] = y_prob_base
                st.session_state["auc_base"] = auc_base
                st.session_state["ap_base"] = ap_base

            st.success("Baseline model trained successfully!")

        if st.session_state["auc_base"] is not None:
            st.markdown("**Baseline Model Performance:**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": ["AUC", "Average Precision"],
                        "Score": [f"{st.session_state['auc_base']:.4f}", f"{st.session_state['ap_base']:.4f}"],
                    }
                )
            )
            st.markdown(
                """
These values become Alex’s **reference point**.
Any augmentation strategy must demonstrate not only improved metrics, but also defensible data quality and privacy.
"""
            )
        else:
            st.info("Click the button to train the baseline model.")


# -----------------------------
# Page 3: SMOTE Augmentation
# -----------------------------
elif st.session_state["page"] == "SMOTE Augmentation":
    st.header("3. Classical Augmentation: SMOTE and its Limitations")

    st.markdown(
        """
A common first attempt to address class imbalance is **SMOTE** (Synthetic Minority Over-sampling Technique).
SMOTE generates synthetic minority points by interpolating between minority neighbors.

Alex knows SMOTE is operationally simple—but in credit data it can be limited because:
- It may not capture complex, non-linear feature relationships.
- It offers **no privacy guarantee** (points are explicit mixtures of real records).
"""
    )

    st.markdown(r"**Mathematical Formulation for SMOTE:**")
    st.markdown(
        r"""
SMOTE generates a new point $\tilde{x}$ by linearly interpolating between a minority sample $x_i$ and one of its
$k$-nearest minority neighbors $x_j$:

$$ 
\tilde{x} = x_i + \lambda (x_j - x_i) 
$$

where $\lambda \sim U(0,1)$.
"""
    )
    st.markdown(
        r"""
Geometrically, SMOTE points lie on line segments between real samples (within the minority class “convex hull”),
which can constrain diversity and does not address privacy.
"""
    )

    if st.session_state["auc_base"] is None:
        st.warning("Please train the baseline model first from 'Baseline Model'.")
    else:
        if st.button("3. Apply SMOTE & Train XGBoost"):
            with st.spinner("Applying SMOTE and training model..."):
                model_smote, y_prob_smote, auc_smote, ap_smote, X_smote_resampled, y_smote_resampled = (
                    apply_smote_and_train_xgboost(
                        st.session_state["X_train_scaled"],
                        st.session_state["y_train"],
                        st.session_state["X_test_scaled"],
                        st.session_state["y_test"],
                    )
                )

                st.session_state["model_smote"] = model_smote
                st.session_state["y_prob_smote"] = y_prob_smote
                st.session_state["auc_smote"] = auc_smote
                st.session_state["ap_smote"] = ap_smote
                st.session_state["X_smote_resampled"] = X_smote_resampled
                st.session_state["y_smote_resampled"] = y_smote_resampled

                # Attempt to isolate newly generated SMOTE samples for privacy comparisons
                # (Assumes first part equals original; SMOTE appends new records)
                try:
                    n_orig = len(st.session_state["X_train_scaled"])
                    st.session_state["X_smote_new_scaled"] = X_smote_resampled.iloc[n_orig:].copy(
                    )
                except Exception:
                    st.session_state["X_smote_new_scaled"] = None

            st.success("SMOTE augmentation and model training successful!")

        if st.session_state["auc_smote"] is not None:
            st.markdown("**SMOTE Augmented Model Performance:**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": ["AUC", "Average Precision"],
                        "Score": [f"{st.session_state['auc_smote']:.4f}", f"{st.session_state['ap_smote']:.4f}"],
                    }
                )
            )
            st.markdown(
                """
Here we see that instead of improving the AUC, SMOTE may have caused a performance drop. This is a common outcome when the synthetic points do not capture the true underlying data distribution, or when they introduce noise that confuses the model. 
Alex must now consider whether the modest performance change (positive or negative) is worth the trade-offs in data quality and privacy.               
SMOTE often improves metrics by balancing training labels, but Alex still needs:
- stronger diversity (non-linear patterns),
- and **privacy defensibility** (especially for data-sharing use cases).
"""
            )
        else:
            st.info("Click the button to apply SMOTE and train.")


# -----------------------------
# Page 4: Conditional VAE
# -----------------------------
elif st.session_state["page"] == "Conditional VAE":
    st.header(
        "4. Advanced Augmentation: Building and Training a Conditional VAE (CVAE)")

    st.markdown(
        """
To overcome SMOTE limitations, Alex evaluates a **Conditional Variational Autoencoder (CVAE)**.
A CVAE learns a probabilistic latent representation of the data **conditioned on the class label**
(default vs non-default). This conditioning enables **targeted generation** of synthetic defaults.

Conceptually:
- The **encoder** learns a distribution over latent variables: $q_\\phi(z|x,y)$
- The **decoder** generates samples from latent space given a label: $p_\\theta(x|z,y)$
- Sampling from the learned latent distribution can produce **novel** records that are statistically consistent,
  without being direct interpolations of real individuals.
"""
    )

    st.markdown(r"**Mathematical Formulation for CVAE:**")
    st.markdown(
        r"""
1) **Encoder (Recognition Model)**

$$ 
q_{\phi}(z|x,y) = \mathcal{N}\left(\mu_{\phi}(x,y), \mathrm{diag}(\sigma_{\phi}^2(x,y))\right) 
$$

2) **Decoder (Generative Model)**

$$ 
p_{\theta}(x|z,y) 
$$

3) **Training Objective (ELBO)**

$$ 
\mathcal{L}(\phi,\theta; x,y) = \|x-\hat{x}\|^2 + \beta\, D_{KL}\left(q_{\phi}(z|x,y)\,\|\,p(z)\right) 
$$

where $p(z)$ is typically a standard normal prior and $D_{KL}$ regularizes the latent space.
"""
    )

    if not st.session_state["df_loaded"]:
        st.warning(
            "Please load and preprocess data first from 'Introduction & Data Overview'.")
    else:
        st.markdown("**CVAE Configuration:**")
        latent_dim_cvae_input = 8

        st.markdown("We are setting the latent dimension to 8, which is a common starting point for tabular data of this size. This allows the CVAE to capture complex relationships without overfitting.")

        st.caption(
            "Higher latent dimension can capture more complexity, but may need more data and careful regularization.")

        if st.button("4. Build & Train CVAE"):
            st.session_state["latent_dim_cvae"] = int(latent_dim_cvae_input)

            with st.spinner("Training CVAE... this can take a few minutes."):
                # Store one-hot labels too (useful for transparency / debugging)
                y_train_oh = to_categorical(
                    st.session_state["y_train"], num_classes=2)
                st.session_state["y_train_oh"] = y_train_oh

                # Use the training helper provided in source.py
                cvae, encoder, decoder, latent_dim_returned = train_cvae(
                    st.session_state["X_train_scaled"],
                    st.session_state["y_train"],
                    latent_dim=st.session_state["latent_dim_cvae"],
                    n_classes=2,
                    epochs=100,
                    batch_size=64,
                    validation_split=0.15,
                )

                st.session_state["cvae_model"] = cvae
                st.session_state["encoder_model"] = encoder
                st.session_state["decoder_model"] = decoder
                st.session_state["latent_dim_cvae"] = int(latent_dim_returned)

            st.success("CVAE trained successfully!")
            st.markdown(
                """
Now Alex can use the **decoder** to generate *new* records conditioned on the **default** class.
This is the foundation for privacy-aware augmentation and for controlled synthetic sharing in validation workflows.
"""
            )
        elif st.session_state["decoder_model"] is not None:
            st.success("CVAE is already trained. Proceed to VAE augmentation.")


# -----------------------------
# Page 5: VAE Augmentation & Model Training
# -----------------------------
elif st.session_state["page"] == "VAE Augmentation & Model Training":
    st.header("5. VAE-Augmented Modeling and Initial Performance Check")

    st.markdown(
        """
With a trained CVAE, Alex can now generate synthetic default records and augment the training set.

A practical risk-management rule applies here: **more synthetic data is not always better**.
Over-generation can inject noise or distort decision boundaries, so Alex caps the number of synthetic defaults
to a reasonable multiple of real defaults (e.g., 5×) to preserve signal integrity.
"""
    )

    if st.session_state["decoder_model"] is None:
        st.warning(
            "Please build and train the CVAE first from 'Conditional VAE'.")
    else:
        st.subheader("Synthetic Data Generation Configuration")
        max_synthetic_ratio_input = st.slider(
            "Max Synthetic-to-Real Ratio for Defaults",
            min_value=1,
            max_value=10,
            value=5,
            key="max_synthetic_ratio",
        )

        if st.button("5. Generate VAE Data & Train XGBoost"):
            with st.spinner("Generating synthetic defaults and training VAE-augmented model..."):
                model_vae, y_prob_vae, auc_vae, ap_vae, X_vae_aug, y_vae_aug, synth_defaults_df = augment_with_vae_and_train_xgboost(
                    st.session_state["X_train_scaled"],
                    st.session_state["y_train"],
                    st.session_state["X_test_scaled"],
                    st.session_state["y_test"],
                    st.session_state["defaults_train"],
                    st.session_state["non_defaults_train"],
                    st.session_state["decoder_model"],
                    st.session_state["latent_dim_cvae"],
                    max_synthetic_ratio=max_synthetic_ratio_input,
                )

                st.session_state["model_vae"] = model_vae
                st.session_state["y_prob_vae"] = y_prob_vae
                st.session_state["auc_vae"] = auc_vae
                st.session_state["ap_vae"] = ap_vae
                st.session_state["X_vae_augmented"] = X_vae_aug
                st.session_state["y_vae_augmented"] = y_vae_aug
                st.session_state["synth_defaults_df"] = synth_defaults_df

            st.success("VAE augmentation and model training successful!")

        if st.session_state["auc_vae"] is not None:
            st.markdown("**VAE Augmented Model Performance:**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": ["AUC", "Average Precision"],
                        "Score": [f"{st.session_state['auc_vae']:.4f}", f"{st.session_state['ap_vae']:.4f}"],
                    }
                )
            )
            st.markdown(
                """
A performance lift is promising, but Alex cannot stop here.
He must validate **synthetic data quality**, **utility**, and **privacy** before making any strategic recommendation.
"""
            )
        else:
            st.info(
                "Click the button to generate VAE data and train the augmented model.")


# -----------------------------
# Page 6: Synthetic Data Assessment
# -----------------------------
elif st.session_state["page"] == "Synthetic Data Assessment":
    st.header(
        "6. Comprehensive Synthetic Data Assessment: Quality, Utility, and Privacy")

    st.markdown(
        """
Apex Financial can only adopt synthetic data if Alex can defend it on three dimensions:

1) **Quality** — Does synthetic data match real statistical properties?
2) **Utility** — Do models trained on synthetic data generalize to real data?
3) **Privacy** — Are synthetic records sufficiently distinct from real individuals?

This section implements the standard toolkit Alex would present to model risk management and compliance reviewers.
"""
    )

    prereq_ok = (st.session_state["auc_vae"] is not None) and (
        st.session_state["synth_defaults_df"] is not None)

    if not prereq_ok:
        st.warning("Please complete 'VAE Augmentation & Model Training' first.")
    else:
        if st.button("6. Perform Comprehensive Assessment"):
            with st.spinner("Running assessment (TSTR, KS, correlations, privacy, visualization)..."):
                # Inverse transform to raw scale for interpretability in distribution/correlation comparisons
                synth_inv = st.session_state["scaler"].inverse_transform(
                    st.session_state["synth_defaults_df"])
                real_inv = st.session_state["scaler"].inverse_transform(
                    st.session_state["defaults_train"])

                synth_defaults_inv_df = pd.DataFrame(
                    synth_inv, columns=st.session_state["X_train_scaled"].columns)
                real_defaults_inv_df = pd.DataFrame(
                    real_inv, columns=st.session_state["X_train_scaled"].columns)

                st.session_state["real_defaults_inv_df"] = real_defaults_inv_df
                st.session_state["synth_defaults_inv_df"] = synth_defaults_inv_df

                # 7.1 TSTR
                st.subheader(
                    "7.1 Train-on-Synthetic, Test-on-Real (TSTR) Protocol")
                st.markdown(
                    r"""
A model is trained **exclusively** on synthetic data and tested on **real** holdout data:

$$ 
TSTR~Ratio = \frac{AUC_{trained~on~synthetic}}{AUC_{trained~on~real}} 
$$

A practical target is often **> 80%** for good synthetic utility (context-dependent).
"""
                )

                tstr_auc, y_prob_tstr, tstr_ratio = run_tstr_protocol(
                    st.session_state["X_train_scaled"],
                    st.session_state["y_train"],
                    st.session_state["X_test_scaled"],
                    st.session_state["y_test"],
                    st.session_state["decoder_model"],
                    st.session_state["latent_dim_cvae"],
                    st.session_state["auc_base"],
                    n_classes=2,
                )

                st.session_state["tstr_auc_value"] = tstr_auc
                st.session_state["tstr_ratio_value"] = tstr_ratio

                st.markdown(f"**TSTR AUC:** `{tstr_auc:.4f}`")
                st.markdown(
                    f"**Baseline AUC:** `{st.session_state['auc_base']:.4f}`")
                if tstr_ratio is not None:
                    st.markdown(f"**TSTR Ratio:** `{tstr_ratio:.2%}`")

                # 7.2 Distribution comparison (KS + overlays)
                st.subheader(
                    "7.2 Feature Distribution Comparison (KS Test & Overlays)")
                key_features = ['utilization', 'income',
                                'debt_ratio', 'past_due_30']
                ks_results, fig_overlay = perform_distribution_comparison(
                    real_defaults_inv_df, synth_defaults_inv_df, key_features)
                st.session_state["ks_test_results"] = ks_results
                # st.dataframe(pd.DataFrame(ks_results))
                if fig_overlay is not None:
                    st.pyplot(fig_overlay)
                    plt.close(fig_overlay)

                # 7.3 Correlation comparison (Frobenius + heatmaps)
                st.subheader(
                    "7.3 Correlation Structure Preservation (Frobenius Error & Heatmaps)")
                frob_err, fig_corr = perform_correlation_comparison(
                    real_defaults_inv_df, synth_defaults_inv_df)
                st.session_state["frobenius_error_value"] = frob_err
                st.markdown(r"**Frobenius Error:**")
                st.markdown(
                    r"""
$$ 
\epsilon_{corr} = \|C_{real} - C_{synth}\|_F = \sqrt{\sum_{i,j}(C^{real}_{ij} - C^{synth}_{ij})^2} 
$$
"""
                )
                st.markdown(f"`{frob_err:.4f}` (lower is better)")
                if fig_corr is not None:
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)

                # 7.4 Privacy assessment (NN distance)
                st.subheader(
                    "7.4 Privacy Assessment (Nearest-Neighbor Distance)")
                min_dist_vae, min_dist_smote, fig_priv = perform_privacy_assessment(
                    X_train_scaled=st.session_state["X_train_scaled"],
                    y_train=st.session_state["y_train"],
                    synth_defaults_df=st.session_state["synth_defaults_df"],
                    X_smote_new_scaled=st.session_state["X_smote_new_scaled"],
                )
                st.session_state["min_distances_vae"] = min_dist_vae
                st.session_state["min_distances_smote"] = min_dist_smote

                if min_dist_vae is not None and len(min_dist_vae) > 0:
                    st.markdown(
                        f"Mean NN distance (VAE): `{np.mean(min_dist_vae):.4f}`")
                    st.markdown(
                        f"Min NN distance (VAE): `{np.min(min_dist_vae):.4f}`")

                if min_dist_smote is not None and len(min_dist_smote) > 0:
                    st.markdown(
                        f"Mean NN distance (SMOTE): `{np.mean(min_dist_smote):.4f}`")
                    st.markdown(
                        f"Min NN distance (SMOTE): `{np.min(min_dist_smote):.4f}`")

                if fig_priv is not None:
                    st.pyplot(fig_priv)
                    plt.close(fig_priv)

                # 7.5 t-SNE visualization
                st.subheader(
                    "7.5 t-SNE / PCA Scatter Plot (Real vs Synthetic Defaults)")
                fig_tsne = perform_tsne_visualization(
                    real_defaults_inv_df, synth_defaults_inv_df)
                if fig_tsne is not None:
                    st.pyplot(fig_tsne)
                    plt.close(fig_tsne)

                st.session_state["assessment_ready"] = True

            st.success("Comprehensive assessment completed!")

        if st.session_state["assessment_ready"]:
            st.markdown(
                """
**How Alex interprets these results:**
- **TSTR Ratio:** indicates whether synthetic data preserves predictive signal.
- **KS & overlays:** validate marginal feature similarity.
- **Correlation + Frobenius:** check multivariate structure and dependencies.
- **Nearest-neighbor distances:** quantify privacy risk (SMOTE typically closer by construction).
- **t-SNE:** sanity-check realism (similar region) without perfect overlap (privacy).
"""
            )
        else:
            st.info("Click the button to run the assessment.")


# -----------------------------
# Page 7: Strategic Review & Comparison
# -----------------------------
elif st.session_state["page"] == "Strategic Review & Comparison":
    st.header(
        "7. Strategic Review: Performance, Regulatory Compliance, and Ethical AI")

    st.markdown(
        """
Alex now consolidates results for Apex Financial’s risk committee. The decision is not purely about “best AUC”—
it must also account for **privacy**, **regulatory expectations**, and **ethical risks** (like bias amplification).

This page summarizes:
- Baseline vs SMOTE vs VAE model performance
- Utility evidence (TSTR)
- A practical qualitative comparison of privacy and diversity
- Discussion points aligned with Basel IRB, GDPR/CCPA, and ECOA-style fairness obligations
"""
    )

    prereq_ok = (
        st.session_state["auc_base"] is not None
        and st.session_state["auc_smote"] is not None
        and st.session_state["auc_vae"] is not None
        and st.session_state["y_prob_base"] is not None
        and st.session_state["y_prob_smote"] is not None
        and st.session_state["y_prob_vae"] is not None
    )

    if not prereq_ok:
        st.warning(
            "Please complete Baseline, SMOTE, and VAE training steps first.")
    else:
        if st.button("7. Consolidate & Review Results"):
            st.subheader("Augmentation Comparison Summary")

            comparison_data = pd.DataFrame(
                {
                    "Method": ["No Augmentation (Baseline)", "SMOTE Augmentation", "VAE Augmentation"],
                    "AUC": [st.session_state["auc_base"], st.session_state["auc_smote"], st.session_state["auc_vae"]],
                    "Average Precision": [st.session_state["ap_base"], st.session_state["ap_smote"], st.session_state["ap_vae"]],
                    "Privacy": ["N/A (real only)", "Low (Interpolation Risk)", "Moderate (Distribution Sampling)"],
                    "Diversity": ["N/A", "Low (Convex Hull)", "High (Learned Distribution)"],
                }
            )
            st.dataframe(comparison_data)

            if st.session_state["tstr_ratio_value"] is not None:
                st.markdown(
                    f"**TSTR Ratio (VAE synthetic → real test):** `{st.session_state['tstr_ratio_value']:.2%}`")

            st.subheader("Performance Comparison (AUC & Average Precision)")
            fig = plt.figure(figsize=(12, 7))
            bar_width = 0.35
            idx = np.arange(len(comparison_data))
            plt.bar(idx, comparison_data["AUC"],
                    bar_width, label="AUC", alpha=0.85)
            plt.bar(idx + bar_width, comparison_data["Average Precision"],
                    bar_width, label="Average Precision", alpha=0.85)
            plt.xlabel("Augmentation Strategy")
            plt.ylabel("Score")
            plt.title(
                "Credit Default Model Performance: Augmentation Strategy Comparison")
            plt.xticks(idx + bar_width / 2,
                       comparison_data["Method"], rotation=12, ha="right")
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader(
                "Discussion Points for Financial Professionals (Alex’s Key Takeaways)")
            st.markdown(
                """
1. **Performance & Utility:** VAE augmentation often improves AUC/AP, and TSTR provides a utility sanity check.
2. **Privacy Preservation:** VAE samples can be meaningfully farther from real records than SMOTE (important for GDPR/CCPA-style privacy expectations).
3. **Data Quality:** Distribution tests and correlation preservation support statistical fidelity.
4. **Regulatory Scrutiny:** Synthetic data is typically best used as a **supplement** (augmentation, validation, collaboration)—not a total replacement for real development data.
5. **Ethical Considerations (Fairness Amplification Risk):** If real data contains historic bias, a generator can reproduce/amplify it. Bias audits must cover **real data, synthetic data, and the resulting models**, consistent with fair-lending expectations.
6. **Production Readiness:** Consider robust libraries (e.g., SDV/CTGAN-family) when mixed types and governance controls matter—balanced against transparency needs.
"""
            )

            st.markdown(
                """
**Bottom line:** CVAEs provide a principled way to address class imbalance while improving privacy posture—**if**
the institution adopts strong validation, bias monitoring, and governance around synthetic generation and use.
"""
            )
        else:
            st.info("Click the button to consolidate and review all results.")


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.write("© 2026 QuantUniversity. All Rights Reserved.")
st.caption(
    "The purpose of this demonstration is solely for educational use and illustration. "
    "All analysis and final editorial judgment are intended for learning and discussion."
)
