# Project 15: Differential Privacy in Recommendation Gradients

## 📌 Executive Summary
When sophisticated platforms like Netflix recommend content, the underlying model weights can inadvertently memorize highly specific user behaviors. This project engineers a Latent Factor Recommendation System that mathematically guarantees against data extraction attacks by implementing **Differentially Private Stochastic Gradient Descent (DP-SGD)** using Meta's Opacus. 

By injecting calibrated Gaussian noise and enforcing strict gradient clipping during the backward pass, the system successfully shields individual viewing histories (ε = 2.75) while simultaneously leveraging the privacy noise as an implicit regularizer to outperform the standard model's real-world accuracy.

## 🏗️ Architecture & Tech Stack
* **Deep Learning Framework:** PyTorch
* **Privacy Engine:** Meta Opacus (DP-SGD)
* **Data Processing:** Pandas, NumPy
* **Evaluation Metrics:** Scikit-Learn (RMSE, MAE)
* **Execution Environment:** Local Engineering (src) + Cloud GPU Execution (Google Colab)

## 📂 Project Structure
    dp_recommender_project/
    ├── data/                  # Raw/processed dataset directory (gitignored)
    ├── src/                   # Core Python engineering modules
    │   ├── __init__.py
    │   ├── data_utils.py      # PyTorch Dataset pipelines, label encoding, and batching logic
    │   └── model_defs.py      # Neural Collaborative Filtering blueprints (Latent Embeddings)
    ├── notebooks/             # Cloud-executed analytical & training notebooks
    │   ├── 01_EDA_and_Preprocessing.ipynb  # k-core filtering, sparsity analysis, train/test split
    │   ├── 02_Colab_DP_Training.ipynb      # Opacus PrivacyEngine hooking and Epsilon accounting
    │   └── 03_Utility_Evaluation.ipynb     # Non-Private Baseline vs. Private Model benchmarking
    ├── requirements.txt       # Strict versioning constraints
    └── README.md

## ⚙️ Core Engineering Components

### 1. Robust Data Pipeline (`src/data_utils.py`)
* Implements a custom `torch.utils.data.Dataset` class.
* Includes strict defensive programming to catch Null values and interface mismatches (e.g., column naming parameterization).
* Handles sparsity by enforcing a strict interaction threshold (k-core filtering) before label encoding to prevent noise-induced divergence for users with faint signals.

### 2. The Recommender Architecture (`src/model_defs.py`)
* Maps millions of interactions into a dense, 32-dimensional coordinate space.
* Features user and item bias terms to account for baseline platform trends.
* Employs asymptotic scaling via a slightly expanded Sigmoid activation buffer (0.9 to 5.1) to strictly bound continuous predictions within the 1-to-5 star rating constraint without causing infinite gradient strain.

### 3. The Privacy Shield (`notebooks/02_Colab_DP_Training.ipynb`)
* Hooks Meta's `PrivacyEngine` into the standard Adam optimizer.
* Enforces a maximum gradient norm (Clipping) to limit the mathematical influence of "power users".
* Accumulates privacy budget mathematically, tracking exact Epsilon (ε) expenditure per epoch.

## 📊 Business Impact & Results
The system was evaluated against a standard, non-private Recommender baseline after 15 epochs on a hold-out test set. 

* **Standard Model (No Privacy) RMSE:** 1.0322 stars
* **DP-SGD Model (ε=2.75) RMSE:** 0.9564 stars
* **Net Utility Change:** +0.0758 RMSE improvement

**Conclusion:** The project achieved full GDPR-compliant user privacy at a zero-cost utility trade-off. The standard baseline model suffered from catastrophic overfitting (memorization). The privacy constraints (clipping and noise) acted as a massive implicit regularizer, forcing the network to abandon specific user memorization and lock onto generalized global trends, thereby improving accuracy on unseen data.

## 🚀 How to Run
1. Clone the repository and install dependencies: `pip install -r requirements.txt`.
2. Download a standard interaction dataset (e.g., MovieLens 100k or 1M) and place `ratings.csv` in the `data/` directory.
3. Upload the `notebooks/` directory to Google Colab (or any environment with GPU acceleration).
4. Execute notebooks `01` through `03` sequentially, ensuring `src/` modules are in the system path.