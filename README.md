# Logistic Regression on Imbalanced Data with Focal Loss

This project implements and evaluates a **Logistic Regression** model for binary classification on imbalanced datasets. It includes a **Baseline Logistic Regression** model and a **Focal Loss variant** to address class imbalance. The project systematically analyzes the performance of both models across multiple datasets using various metrics and visualizations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Methodology](#methodology)
7. [Results and Insights](#results-and-insights)
8. [Limitations and Future Work](#limitations-and-future-work)
9. [References](#references)

---

## Project Overview

Class imbalance is a common challenge in machine learning, where one class significantly outnumbers the other. This project explores the limitations of standard Logistic Regression in such scenarios and proposes a Focal Loss variant to improve minority class detection.

### Key Objectives:
- Implement a **Baseline Logistic Regression** model from scratch.
- Develop a **Focal Loss variant** to address class imbalance.
- Evaluate both models on multiple imbalanced datasets.
- Compare performance using metrics like **F1-score**, **Precision**, **Recall**, **ROC AUC**, and **PR AUC**.
- Provide detailed visualizations and statistical analyses.

---

## Features

- **Custom Logistic Regression Implementation**:
  - Gradient descent optimization.
  - Support for regularization (L1, L2).
  - Binary cross-entropy loss.

- **Focal Loss Variant**:
  - Implements Focal Loss to focus on hard-to-classify examples.
  - Tunable `gamma` parameter for controlling the focus effect.

- **Comprehensive Evaluation**:
  - Metrics: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, MCC, BACC.
  - Visualizations: Confusion matrices, boxplots, radar plots, scatter plots, and heatmaps.
  - Statistical tests: Wilcoxon signed-rank test, rank-biserial correlation.

- **Dataset Analysis**:
  - Handles multiple imbalanced datasets.
  - Preprocessing: Binary target column detection, one-hot encoding, label encoding.
  - Visualizations: Class imbalance distribution, class frequencies.

---

## Installation

### Prerequisites:
- Python 3.8 or higher
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `autograd`, `plotly`

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/logistic-regression-imbalanced.git
   cd logistic-regression-imbalanced


---


## Usage

Running the Notebook:

1. Open the Jupyter Notebook: jupyter notebook notebook.ipynb

2. Follow the sections in the notebook:

- Baseline Logistic Regression: Implementation, training, and evaluation.
- Focal Loss Variant: Implementation, gamma tuning, and evaluation.
- Comparative Analysis: Metrics, visualizations, and statistical insights.

3. Key Sections:

- Data Loading and Preprocessing:
    - Automatically discovers and preprocesses datasets in the class_imbalance/ folder.
- Baseline Model:
    - Trains and evaluates the Logistic Regression model.
- Focal Loss Model:
    - Tunes the gamma parameter and evaluates the Focal Loss variant.
- Visualization and Analysis:
    - Generates detailed plots and statistical summaries.


## Project Structure

├── class_imbalance/          # Folder containing CSV datasets
├── [notebook.ipynb](http://_vscodecontentref_/0)            # Main Jupyter Notebook
├── README.md                 # Project documentation
└── Presentation.pdf  # slides focusing on the main issues of the assignment



## Methodology

1. Baseline Logistic Regression:

- Implements Logistic Regression from scratch using gradient descent.
- Optimizes binary cross-entropy loss.
- Evaluates performance on imbalanced datasets.

2. Focal Loss Variant:

- Modifies the loss function to focus on hard-to-classify examples.
- Introduces the gamma parameter to control the focus effect.
- Tunes gamma using cross-validation.

3. Evaluation Metrics:

- Accuracy: Overall correctness of predictions.
- Precision: Proportion of true positives among predicted positives.
- Recall: Proportion of true positives among actual positives.
- F1-score: Harmonic mean of Precision and Recall.
- ROC AUC: Area under the ROC curve.
- PR AUC: Area under the Precision-Recall curve.
- MCC: Matthews Correlation Coefficient.
- BACC: Balanced Accuracy.

4. Statistical Analysis:

- Wilcoxon signed-rank test for metric significance.
- Rank-biserial correlation for effect size.


## Methodology

1. Baseline Logistic Regression:

Implements Logistic Regression from scratch using gradient descent.
Optimizes binary cross-entropy loss.
Evaluates performance on imbalanced datasets.

2. Focal Loss Variant:

- Modifies the loss function to focus on hard-to-classify examples.
- Introduces the gamma parameter to control the focus effect.
- Tunes gamma using cross-validation.

3. Evaluation Metrics:

- Accuracy: Overall correctness of predictions.
- Precision: Proportion of true positives among predicted positives.
- Recall: Proportion of true positives among actual positives.
- F1-score: Harmonic mean of Precision and Recall.
- ROC AUC: Area under the ROC curve.
- PR AUC: Area under the Precision-Recall curve.
- MCC: Matthews Correlation Coefficient.
- BACC: Balanced Accuracy.

4. Statistical Analysis:

- Wilcoxon signed-rank test for metric significance.
- Rank-biserial correlation for effect size.

## Limitations and Future Work

Limitations:

- Focal Loss does not guarantee improvement for all datasets or metrics.
- Gains in metrics like ROC AUC and PR AUC are inconsistent.
- Optimal gamma value varies across datasets, requiring careful tuning.


## Future Work:

- Explore ensemble methods or hybrid approaches.
- Investigate other loss functions (e.g., Dice Loss, Tversky Loss).



## References


1. Focal Loss Paper: Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*. [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

2. Scikit-learn Documentation: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. [https://scikit-learn.org/](https://scikit-learn.org/)

3. Autograd Library: Maclaurin, D., Duvenaud, D., & Adams, R. (2015). Autograd: Effortless Gradients in Numpy. [https://github.com/HIPS/autograd](https://github.com/HIPS/autograd)

4. Imbalanced Classification Techniques: He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284. [https://doi.org/10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239)

5. Evaluation Metrics for Imbalanced Data: Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*, 10(3), e0118432. [https://doi.org/10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432)

6. Python Libraries:
   - Matplotlib: Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90-95. [https://matplotlib.org/](https://matplotlib.org/)
   - Pandas: McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56. [https://pandas.pydata.org/](https://pandas.pydata.org/)
   - Seaborn: Waskom, M. L. (2021). Seaborn: Statistical Data Visualization. *Journal of Open Source Software*, 6(60), 3021. [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

7. Class Imbalance Visualization: Fernández, A., García, S., Galar, M., Prati, R. C., Krawczyk, B., & Herrera, F. (2018). Learning from Imbalanced Data Sets. *Springer International Publishing*. [https://doi.org/10.1007/978-3-319-98074-4](https://doi.org/10.1007/978-3-319-98074-4)

8. Custom Logistic Regression Implementation:[https://github.com/rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms)

9. Wilcoxon Signed-Rank Test: Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), 80-83. [https://doi.org/10.2307/3001968](https://doi.org/10.2307/3001968)
