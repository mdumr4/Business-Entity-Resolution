# Master Blueprint: Engineering Multimodal AI Solutions for High-Stakes ML Challenges

## Executive Summary & System Overview

When tackling high-stakes, multimodal competitive machine learning challenges—such as the **Amazon ML Challenge 2025 (Smart Product Pricing Challenge)**—the primary technical difficulty lies in extracting, aligning, and fusing heterogeneous information signals (unstructured catalog text, raw image URLs, and categorical/numerical metadata) into a unified predictive system. 

This document synthesizes the exact engineering methodologies, architectural paradigms, feature extraction pipelines, custom loss formulations, and ensembling strategies developed by top-ranking teams (including **Team SPAM_LLMs - AIR 3/5**, **Team ML Mavericks - AIR 80 / Top 0.3%**, **Team LORAverse - AIR 377**, and **SmartPricingChallenge**). Beyond detailing *what* was built, this guide provides a step-by-step meta-prompting framework on *how* to instruct an AI system or engineering team to solve similar multimodal tasks from scratch.

---

## 1. Challenge & Domain Problem Framework

### 1.1 Problem Statement
The objective is to predict continuous product prices ($y \in \mathbb{R}^+$) using strictly provided multimodal e-commerce catalog data:
* **Catalog Content Text**: A single text field containing product title, detailed description, bullet points, and Item Pack Quantity (IPQ) concatenated together.
* **Product Image Links**: Public URLs pointing to high-resolution product catalog images.
* **Key Constraint**: **Zero external price lookups or web scraping allowed**. Models must infer pricing strictly from intrinsic semantic, aesthetic, and structural features.

### 1.2 Evaluation Metric & Target Dynamics
* **Evaluation Metric**: Symmetric Mean Absolute Percentage Error (SMAPE), defined as:
  $$\text{SMAPE} = \frac{100}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{\frac{|y_i| + |\hat{y}_i|}{2}}$$
  * Range: $0\%$ to $200\%$ (lower is better).
  * SMAPE heavily penalizes relative over-predictions and scale mismatches.
* **Distributional Skew**: E-commerce pricing exhibits extreme right-skewness spanning multiple orders of magnitude (e.g., \$0.99 items to \$5,000+ luxury electronics).
* **Target Scaling**: Applying a logarithmic transformation to the target variable:
  $$z_i = \log(y_i + 1) \quad \Longleftrightarrow \quad y_i = \exp(z_i) - 1$$
  This stabilizes variance, prevents large targets from dominating gradients, and naturally converts absolute error minimization into relative percentage error optimization.

---

## 2. Deep Dive: Architectural Paradigms of Top Candidates

---

### 2.1 Candidate 1: Team SPAM_LLMs (AIR 3 Public / AIR 5 Private)
* **Core Philosophy**: *Large Frozen Foundation Encoders > Fine-Tuned Small Distilled Encoders*.
* **Key Insight**: Distilling Large Language Models (LLMs) down to smaller parameters severely degrades internal vector representation capabilities. Fine-tuning smaller models often hits a performance ceiling quickly, whereas extracting embeddings directly from massive, frozen foundation models yields far richer semantic geometry.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CATALOG INPUT DATA                               │
│  • Catalog Text (Title, Bullet Points, Description, Quantity)               │
│  • Product Image URL                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
 ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
 │    TEXT EMBEDDINGS  │    │  MULTIMODAL EMBEDS  │    │   VISION EMBEDDINGS │
 │     Qwen-3-4B       │    │   SigLIP 2 Giant    │    │    DINOv3 Giant     │
 │  (Frozen Foundation)│    │     (2B Params)     │    │    (0.8B Params)    │
 └─────────────────────┘    └─────────────────────┘    └─────────────────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                       MODALITY-SPECIFIC NETWORKS                          │
 │  • Separate Linear Projection Layers mapping distinct vector spaces       │
 └───────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                       GATED REGRESSOR NETWORK                             │
 │  • Trained with Custom Log-based MSE Loss                                 │
 └───────────────────────────────────────────────────────────────────────────┘
```

#### Detailed Workflow & Preprocessing
1. **Catalog Text Cleaning**:
   * **Text Truncation**: To fit context windows without injecting noise, if `Product Description` exists, use it exclusively; otherwise, extract only the top 5 `Bullet Points`.
   * **Unit Normalization**: Map inconsistent unit tokens to standardized vocabulary representations (e.g., `gm`, `g`, `grams` $\rightarrow$ `grams`).
   * **Number-to-Word Conversion**: Convert raw digits to text representation (e.g., `100` $\rightarrow$ `One Hundred`). This significantly aligns numeric values with LLM tokenizer semantic space.
2. **Embedding Pipeline**:
   * **Text Modality**: Qwen-3-4B (4 Billion parameters).
   * **Vision-Language Modality**: Google SigLIP 2 Giant (`siglip2-giant-opt-patch16-256`, 2B parameters).
   * **Pure Vision Modality**: Meta DINOv3 Giant (`dinov3-vith16plus-pretrain-lvd1689m`, 0.8B parameters).
3. **Modality-Specific Transformation Networks**:
   * Direct concatenation of raw vectors from different models fails because each model projects features into a distinct high-dimensional manifold.
   * Dedicated linear/dense projection heads transform each raw embedding stream into an aligned latent dimension prior to fusion.
4. **Custom Loss Function (Log-MSE)**:
   $$\mathcal{L}_{\text{Log-MSE}} = \frac{1}{n} \sum_{i=1}^{n} \left( \log(y_i + 1) - \hat{y}_i \right)^2$$
   Where $\hat{y}_i$ is the model's direct prediction in the log-transformed space.

---

### 2.2 Candidate 2: Team LORAverse (AIR 377 / 82,787+ Participants)
* **Core Philosophy**: *Supervised Fine-Tuning on Primary Signal + Domain Image Encoder + Gated Residual MLP + Hybrid Tree Ensembling*.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             INPUT SOURCES                                │
│   Product Image   │   Item Name   │   Bullet Points   │   Unit / Value    │
└───────────────────┴───────────────┴───────────────────┴──────────────────┘
          │                 │                 │                   │
          ▼                 ▼                 ▼                   ▼
  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────────────┐
  │ Marqo-B Vision│ │ Fine-tuned    │ │ Fine-tuned    │ │ Learnable 32-dim │
  │ (512-dim)     │ │ Qwen-4B       │ │ Qwen-4B       │ │ Unit Embeddings  │
  └───────────────┘ └───────────────┘ └───────────────┘ └──────────────────┘
          │                 │                 │                   │
          └─────────────────┼─────────────────┴───────────────────┘
                            ▼
         ┌─────────────────────────────────────┐
         │        GATED FUSION LAYER           │
         │  (Dynamic Weights Per Modality)     │
         └─────────────────────────────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────┐
         │         RESIDUAL MLP BLOCK          │
         └─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┬───────────────┐
            ▼               ▼               ▼               ▼
      ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
      │ Deep Regr │   │ LightGBM  │   │  XGBoost  │   │ CatBoost  │
      │   (0.85)  │   │   (0.05)  │   │   (0.05)  │   │   (0.05)  │
      └───────────┘   └───────────┘   └───────────┘   └───────────┘
            │               │               │               │
            └───────────────┼───────────────┴───────────────┘
                            ▼
              ┌───────────────────────────┐
              │   FINAL PRICE PREDICTION  │
              └───────────────────────────┘
```

#### Detailed Workflow & Technical Innovations
1. **Targeted Fine-Tuning**:
   * Fine-tuned a **Qwen-4B variant** using Supervised Fine-Tuning (SFT) explicitly on `(item_name, price)` tuple pairs. This forced the language model to learn domain pricing nuances (e.g., distinguishing "Flagship Smartphone" vs. "Budget Phone Cover").
2. **E-Commerce Visual Feature Extraction**:
   * Utilized **Marqo-B E-Commerce Embeddings** (`marqo-ecommerce-embeddings-B`, 200M parameters), producing a specialized 512-dimensional visual vector optimized for product visual similarity.
3. **Structured Categorical Embedding Head**:
   * Regex parsing separated scalar quantities from units (e.g., "500" and "ml").
   * Passed categorical units through a **32-dimensional learnable embedding matrix**, combining it with linear projections of scalar values.
4. **Gated Residual MLP Fusion Layer**:
   * Instead of static concatenation, a gating network calculates dynamic attention weights for each modality stream:
     * **Text-heavy dynamic weighting** for technical products (electronics, computer hardware).
     * **Vision-heavy dynamic weighting** for aesthetic products (apparel, decor).
5. **Hybrid Weighted Stacking Formula**:
   $$\hat{z}_{\text{final}} = 0.85 \cdot \hat{z}_{\text{mlp}} + 0.05 \cdot \hat{z}_{\text{lgb}} + 0.05 \cdot \hat{z}_{\text{xgb}} + 0.05 \cdot \hat{z}_{\text{cat}}$$
   $$\hat{y}_{\text{final}} = \exp(\hat{z}_{\text{final}}) - 1$$

---

### 2.3 Candidate 3: Team ML Mavericks (AIR 80 / Top 0.3%)
* **Core Philosophy**: *Exhaustive Hypothesis Testing Across 35+ Model Architecture Variations + Diagnostic Gap Remediation*.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EXPLORATORY DATA ANALYSIS                        │
│   • Price classification binning strategies & diagnostic Gap Analysis      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
 ┌────────────────────────────────────┐        ┌──────────────────────────────┐
 │     30+ TEXT-ONLY PIPELINES        │        │    MULTIMODAL EXPERIMENTS    │
 │ • LLMs: Granite 4.0 (Unsloth), Qwen│        │ • CLIP Image + Brand Features│
 │ • Encoders: FLAN-T5, BERT (MMD Loss)│        │ • Qwen2.5 Vision-Language    │
 │ • Classical: GBDTs, LSTM, FAISS    │        │   Model Fine-Tuning          │
 └────────────────────────────────────┘        └──────────────────────────────┘
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                 HYBRID ENSEMBLE WITH VALIDATION-TEST GAP FIXES            │
 └───────────────────────────────────────────────────────────────────────────┘
```

#### Detailed Workflow & Methodological Mastery
1. **Comprehensive Diagnostic Analysis**:
   * Conducted explicit price classification and binning experiments (`classification-idea.ipynb`) to understand pricing clusters.
   * Discovered that local validation scores often drifted from leaderboard evaluations due to imbalance across pricing percentiles.
2. **Iterative Model Exploration**:
   * Built 30+ text-only pipelines spanning traditional GBDTs, LSTMs, T5 with dynamic length handling, BERT with Maximum Mean Discrepancy (MMD) loss, and Granite 4.0 fine-tuned with Unsloth optimization on A100 GPUs.
   * Multi-Modal extension incorporated CLIP-based vision features paired with brand interactions and Qwen2.5 Vision-Language fine-tuning.
3. **Validation & Distribution Gap Fixes**:
   * Implemented **10-Bin Stratified 5-Fold Cross-Validation** based on price percentiles to enforce identical pricing distributions across training and validation splits.
   * Final solution merged high-performing LLM text predictions with multimodal embeddings through a hybrid ensemble explicitly calibrated to eliminate public/private leaderboard drift.

---

### 2.4 Candidate 4: SmartPricingChallenge Pipeline
* **Core Philosophy**: *High-Dimensional Feature Engineering (2,263 Features) + 2-Level Stacking Architecture*.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INPUT: 75,000 CATALOG SAMPLES                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
 ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
 │    TEXT FEATURES    │    │    IMAGE FEATURES   │    │   DOMAIN FEATURES   │
 │      (880-dim)      │    │     (1,333-dim)     │    │      (50-dim)       │
 │ • DistilBERT (768)  │    │ • EfficientNet (1280│    │ • Brand Regex       │
 │ • TF-IDF (100)      │    │ • Color Hists (39)  │    │ • IPQ Parsing       │
 │ • Stat Metrics (12) │    │ • Gabor Texture (8) │    │ • Category Infer    │
 │                     │    │ • Quality (6)       │    │                     │
 └─────────────────────┘    └─────────────────────┘    └─────────────────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                   FEATURE CONCATENATION (2,263 DIMS)                      │
 └───────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                  LEVEL-1 BASE MODELS (5-FOLD OOF)                         │
 │        XGBoost  │  LightGBM  │  CatBoost  │  PyTorch Deep NN             │
 └───────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                  LEVEL-2 META-LEARNER (RIDGE REGRESSION)                  │
 └───────────────────────────────────────────────────────────────────────────┘
```

#### Detailed Feature Breakdown (2,263 Dimensions)
1. **Text Modality (880 dimensions)**:
   * **DistilBERT Transformer Embeddings** (768-dim): Contextual semantic representation.
   * **TF-IDF N-grams (1,3)** (100-dim): Specific keyword, brand name, and specification indicators.
   * **Statistical Text Metrics** (12-dim): Text length, word counts, digit-to-letter ratio, uppercase character frequency, special character density.
2. **Vision Modality (1,333 dimensions)**:
   * **EfficientNet-B0 CNN Embeddings** (1,280-dim): High-level visual category representation.
   * **Color Histograms & Dominant Colors** (39-dim): 27-dim RGB color distribution + 9-dim dominant color clustering (K-Means) + 3-dim mean RGB value.
   * **Texture Analysis** (8-dim): Gabor filters applied across 4 orientations and 2 spatial scales to quantify surface texture.
   * **Quality & Technical Metrics** (6-dim): Laplacian variance (image sharpness), contrast, brightness, and aspect ratio.
3. **Domain & Tabular Metadata (50 dimensions)**:
   * Brand regex extraction, Item Pack Quantity (IPQ) scalar counts, and category matching.
4. **Optimization & Performance**:
   * **NumPy Memory-Mapped Feature Caching**: Caching pre-extracted feature arrays on disk reduced model iteration loops from 2 hours to 5 minutes (**24x speedup**), reducing total pipeline training time to 5.6 hours.

---

## 3. Systematic Summary Comparison Matrix

| Component / Dimension | SPAM_LLMs (AIR 3 / 5) | LORAverse (AIR 377) | ML Mavericks (AIR 80) | SmartPricing Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Text Strategy** | Frozen Qwen-3-4B Embeddings | Fine-tuned Qwen-4B on Name/Price | Granite 4.0 / Qwen2.5 / T5 | DistilBERT + TF-IDF (880-dim) |
| **Primary Vision Strategy**| SigLIP 2 Giant + DINOv3 Giant | Marqo-B E-Commerce (512-dim) | CLIP + Qwen2.5 Vision | EfficientNet-B0 + Color/Texture (1333-dim) |
| **Text Cleaning Trick** | Number-to-Words & Unit Norm | Regex Unit Extraction | Comprehensive Diagnostic Parsing | Stat Metrics & Keyword Filters |
| **Modality Fusion** | Modality Projection Networks | Gated Residual MLP | Multimodal Embedding Stacking | Direct Feature Concatenation |
| **Loss Function** | Custom Log-based MSE | MSE on $\log(\text{price})$ | Log MSE & SMAPE Alignment | MSE on Log Scale |
| **Ensembling Method** | Gated Regressor Head | Weighted Average (0.85 MLP + Boosters)| Hybrid Stacking & Gap Fixes | Level-2 Ridge Stacking on OOF |
| **Key Advantage** | High representation richness | Dynamic modality gating | Diagnostic validation stability | Explicit feature engineering depth |

---

## 4. Meta-Prompt Framework: How to Instruct an AI to Solve Multimodal ML Challenges

To prompt or instruct an AI system (or team) to solve a complex multimodal ML task, execute the following structured 6-phase pipeline:

```markdown
### PHASE 1: DATA EXPLORATION & TARGET TRANSFORMATION
1. Analyze target variable distribution. If right-skewed, apply log1p transformation: z = log(y + 1).
2. Establish 5-Fold Stratified K-Fold Cross Validation binned across 10 target percentile bins.
3. Clean text: Standardize unit variations, convert numbers to words, and parse regex entities (e.g. IPQ).

### PHASE 2: FEATURE EXTRACTION PIPELINE
1. Text Encoders: Extract embeddings from a large frozen LLM encoder (e.g., Qwen-4B / T5) and compute TF-IDF n-grams.
2. Vision Encoders: Extract embeddings using specialized domain vision models (SigLIP2 / DINOv3 / Marqo-B) alongside color and texture metrics.
3. Precompute & Cache: Save all extracted vector matrices to memory-mapped files (.npy) to accelerate training.

### PHASE 3: MODALITY ALIGNMENT & FUSION
1. Pass raw embeddings through Modality-Specific Linear Projection layers to map them into an aligned latent dimension.
2. Implement a Gated Residual MLP Fusion Layer that dynamically learns attention weights across text, vision, and tabular inputs.

### PHASE 4: LOSS FUNCTION & REGRESSION HEAD
1. Train neural regression heads using Log-based MSE loss: L = (1/n) * sum((log(y_i + 1) - y_hat_i)^2).
2. Align loss function updates with evaluation metrics (e.g., SMAPE / MAPE).

### PHASE 5: BASE MODELING & HYBRID ENSEMBLING
1. Train Level-1 models: PyTorch Gated MLP + XGBoost + LightGBM + CatBoost on 5-Fold Out-of-Fold (OOF) splits.
2. Combine predictions via Level-2 Meta-Learner (Ridge Regression) or weighted averaging (e.g., 0.85 Neural Net + 0.15 GBDTs).

### PHASE 6: POST-PROCESSING & VALIDATION AUDIT
1. Transform predictions back to original space: y_pred = exp(z_pred) - 1. Enforce non-negativity clipping.
2. Audit validation-to-leaderboard drift before committing final predictions.
```
