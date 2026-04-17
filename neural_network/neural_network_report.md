# Neural Network Report Draft

## Task 4. Neural Networks

### 4.1 Additional Processing

| Processing step | Variables affected | Purpose | Notes |
|---|---|---|---|
| One-hot encoding | `Agriblock`, `Variety`, `Soil Types`, `Nursery`, wind direction variables | Convert categorical variables into numeric dummy variables | Required because `MLPClassifier` cannot use text labels directly |
| Median imputation | `Min temp_D1_D30`, `Min temp_D31_D60`, `Min temp_D61_D90`, `Min temp_D91_D120` | Fill missing numeric values | Median was used because it is less sensitive to outliers than the mean |
| Standardisation | Continuous numeric variables | Put numeric inputs on a comparable scale | Required because neural networks are sensitive to feature scale |
| Keep binary variables as 0/1 | One-hot encoded dummy variables | Preserve binary encoded meaning | These variables are already on a consistent scale |
| Train/test split | Full processed dataset | Separate model training from final testing | Current split: 75% training and 25% testing, using `stratify=y` |

### 4.2 Full Neural Network Model

| Item | Current value |
|---|---|
| Model | `MLPClassifier` |
| Input feature set | Full feature set after preprocessing and high-correlation removal |
| Number of input variables | 50 |
| Train/test split | 75% / 25% |
| Cross-validation | 5-fold CV |
| Scoring metric for GridSearchCV | Accuracy |
| Best hidden layer architecture | `(50, 25)` |
| Best activation | `relu` |
| Best alpha | `0.0001` |
| Best learning rate init | `0.01` |
| Best CV accuracy | `0.8919` |
| Training accuracy | `0.9102` |
| Test accuracy | `0.8904` |
| Train-test accuracy gap | `0.0198` |
| AUC | `0.9425` |
| Iterations used | 43 |
| Convergence / stopping | Stopped before `max_iter=1000` with early stopping |

Hyperparameter search range:

| Hyperparameter | Search values | Reason |
|---|---|---|
| `hidden_layer_sizes` | `(25,)`, `(50,)`, `(50, 25)` | Tests one small layer, one medium layer, and a two-layer architecture |
| `activation` | `relu`, `tanh` | Common activation functions for MLP models |
| `alpha` | `0.0001`, `0.001`, `0.01` | Tests different L2 regularisation strengths |
| `learning_rate_init` | `0.001`, `0.01` | Tests stable default learning rate and a faster larger learning rate |

### 4.3 Reduced Feature Neural Network Model

Feature selection method:

| Item | Description |
|---|---|
| Feature selection method | Tuned decision tree feature importance |
| Decision tree criterion | `gini` |
| Decision tree max depth | 5 |
| Decision tree min samples leaf | 5 |
| Decision tree min samples split | 20 |
| Selection rule | Keep variables with `feature_importances_ > 0` |

Selected variables:

| No. | Variable |
|---:|---|
| 1 | `Seedrate(in Kg)` |
| 2 | `Variety_delux ponni` |
| 3 | `Variety_ponmani` |
| 4 | `Relative Humidity_D31_D60` |
| 5 | `Max temp_D61_D90` |
| 6 | `Wind Direction_D61_D90_NNW` |
| 7 | `Wind Direction_D31_D60_NE` |
| 8 | `51_70DRain(in mm)` |
| 9 | `Wind Direction_D1_D30_NA` |
| 10 | `Wind Direction_D61_D90_SW` |
| 11 | `Nursery_wet` |
| 12 | `Soil Types_clay` |
| 13 | `Wind Direction_D31_D60_W` |

Performance:

| Item | Current value |
|---|---|
| Model | `MLPClassifier` |
| Input feature set | Decision-tree-selected reduced feature set |
| Number of input variables | 13 |
| Train/test split | 75% / 25% |
| Cross-validation | 5-fold CV |
| Scoring metric for GridSearchCV | Accuracy |
| Best hidden layer architecture | `(25, 10)` |
| Best activation | `relu` |
| Best alpha | `0.001` |
| Best learning rate init | `0.01` |
| Best CV accuracy | `0.8914` |
| Training accuracy | `0.9026` |
| Test accuracy | `0.8935` |
| Train-test accuracy gap | `0.0091` |
| AUC | `0.9416` |
| Iterations used | 39 |
| Convergence / stopping | Stopped before `max_iter=1000` with early stopping |

### 4.4 ROC Curve and Model Comparison

Performance comparison:

| Metric | Full neural network | Reduced neural network | Comment |
|---|---:|---:|---|
| Number of input variables | 50 | 13 | Reduced model uses 37 fewer variables |
| Best CV accuracy | 0.8919 | 0.8914 | Almost identical |
| Training accuracy | 0.9102 | 0.9026 | Full model is slightly higher on training data |
| Test accuracy | 0.8904 | 0.8935 | Reduced model is slightly higher |
| Train-test accuracy gap | 0.0198 | 0.0091 | Reduced model has a smaller gap |
| AUC | 0.9425 | 0.9416 | Full model is marginally higher |
| Iterations used | 43 | 39 | Both stopped early |

Draft interpretation:

| Point | Draft note |
|---|---|
| Overall performance | The two neural network models show very similar predictive performance. The differences in test accuracy and AUC are very small. |
| Feature selection outcome | The reduced model uses only 13 variables but achieves similar performance to the full model, suggesting many removed variables were redundant or less informative. |
| Accuracy vs AUC | The reduced model has slightly higher test accuracy, while the full model has a marginally higher AUC. This means neither model is clearly superior on every metric. |
| Overfitting | Both models have small train-test accuracy gaps. The reduced model has the smaller gap, so it shows less overfitting risk in this run. |
| Interpretability | Both neural networks are less interpretable than decision trees or logistic regression, but the reduced model is easier to discuss because it uses fewer input variables. |
| Computational complexity | The reduced model is simpler and should be faster to train and easier to deploy because it requires fewer variables. |
| Current conclusion | If performance remains similar after final tuning, the reduced neural network may be preferable because it keeps predictive performance while reducing complexity. |

Notes to update after parameter tuning:

| Item to check | Update needed |
|---|---|
| Best hyperparameters | Update full and reduced model tables if GridSearchCV selects different values |
| Accuracy values | Update training accuracy, test accuracy, and CV accuracy |
| AUC values | Update ROC/AUC table after re-running ROC cells |
| Selected features | Update if decision tree feature selection changes |
| Final conclusion | Adjust if one model becomes clearly better after tuning |
