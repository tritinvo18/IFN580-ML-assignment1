# Task 4. Neural Networks

## 4.1 Additional Processing for Neural Network Modelling

Neural networks require all input variables to be numeric and are sensitive to the scale of the input features. Therefore, categorical variables were converted into numeric dummy variables, missing numeric values were imputed, and continuous numeric variables were standardised before fitting the `MLPClassifier`. The target variable was `isAboveAvg`, where 1 means the paddy yield per hectare is above the dataset mean and 0 means it is not.

| Processing step | Variables affected | Purpose |
|---|---|---|
| One-hot encoding | `Agriblock`, `Variety`, `Soil Types`, `Nursery`, wind direction variables | Converted categorical inputs into numeric dummy variables |
| Median imputation | `Min temp_D1_D30`, `Min temp_D31_D60`, `Min temp_D61_D90`, `Min temp_D91_D120` | Filled missing numeric values using a robust central value |
| High-correlation removal | Variables with pairwise correlation greater than 0.98 | Removed redundant inputs before modelling |
| Standardisation | Continuous numeric variables | Put numeric inputs on a comparable scale for neural network training |
| Binary dummy handling | One-hot encoded variables | Kept as 0/1 because these variables were already on a consistent scale |
| Train/test split | Full processed dataset | Used a 75% training and 25% testing split with `stratify=y` |

The scaler was fitted on the training data only and then applied to the test data. This avoids data leakage from the test set into the training process.

## 4.2 Full Neural Network Tuned with GridSearchCV

The first neural network used the full processed input set after preprocessing and high-correlation removal. This produced 50 input variables. The model function was `MLPClassifier`, using the `adam` solver, `max_iter=1000`, `early_stopping=True`, and `n_iter_no_change=20`.

GridSearchCV was used to tune the model because neural networks are sensitive to hyperparameters such as hidden-layer size, activation function, regularisation strength, and learning rate. The search used classification accuracy as the scoring metric.

| Hyperparameter | Search values | Reason |
|---|---|---|
| `hidden_layer_sizes` | `(25,)`, `(50,)`, `(50, 25)` | Tested one smaller layer, one medium layer, and a two-layer architecture |
| `activation` | `relu`, `tanh` | Compared two common nonlinear activation functions |
| `alpha` | `0.0001`, `0.001`, `0.01` | Tested different L2 regularisation strengths to control overfitting |
| `learning_rate_init` | `0.001`, `0.01` | Compared a stable default learning rate with a faster larger learning rate |

The main full neural network selected by the 10-fold GridSearchCV used the following settings:

| Item | Value |
|---|---|
| Model | `MLPClassifier` |
| Input variables | 50 |
| Best hidden-layer architecture | `(50, 25)` |
| Best activation | `relu` |
| Best alpha | `0.01` |
| Best learning rate init | `0.01` |
| Best CV accuracy | `0.8909` |
| Training accuracy | `0.9137` |
| Test accuracy | `0.8965` |
| Train-test accuracy gap | `0.0173` |
| AUC | `0.9360` |
| Iterations used | 48 |

The model stopped after 48 iterations, well before `max_iter=1000`, because early stopping was enabled. This indicates that training stopped once validation performance stopped improving. The train-test accuracy gap was 0.0173, so the full neural network did not show strong evidence of overfitting in the main run, although the training accuracy was higher than the test accuracy.

## 4.3 Reduced Feature Neural Network

The second neural network used a reduced feature set selected by the tuned decision tree from Task 2. The tuned decision tree used `criterion='gini'`, `max_depth=5`, `min_samples_leaf=5`, and `min_samples_split=20`. Features with positive decision-tree feature importance were retained. This rule selected 13 input variables.

| No. | Selected input variable |
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

Because the reduced model had fewer input variables, the architecture search was smaller than the full model search. The reduced model tested `(10,)`, `(25,)`, and `(25, 10)` as hidden-layer structures, with the same activation, regularisation, and learning-rate options.

| Item | Value |
|---|---|
| Model | `MLPClassifier` |
| Input variables | 13 |
| Feature selection method | Tuned decision tree feature importance |
| Selection rule | Keep variables with `feature_importances_ > 0` |
| Best hidden-layer architecture | `(25, 10)` |
| Best activation | `relu` |
| Best alpha | `0.001` |
| Best learning rate init | `0.01` |
| Best CV accuracy | `0.8914` |
| Training accuracy | `0.9026` |
| Test accuracy | `0.8935` |
| Train-test accuracy gap | `0.0091` |
| AUC | `0.9416` |
| Iterations used | 39 |

The reduced neural network stopped after 39 iterations, also before reaching `max_iter=1000`. The train-test accuracy gap was 0.0091, which is smaller than the full model gap in the main run. Therefore, the reduced model showed less evidence of overfitting. Feature selection reduced the input space from 50 variables to 13 variables while maintaining similar predictive performance, so it favoured the outcome in terms of simplicity and generalisation stability. However, it did not clearly dominate the full model on every metric because the full model had slightly higher test accuracy in the main run.

## 4.4 ROC Curve and Model Comparison

Several neural network runs were tested to check whether changing the cross-validation setting or expanding the hyperparameter search would materially improve performance. All models used `MLPClassifier` with the `adam` solver, `max_iter=1000`, `early_stopping=True`, `n_iter_no_change=20`, and GridSearchCV scored by accuracy.

| Run | Model | Inputs | Architecture | Activation | Alpha | CV acc. | Train acc. | Test acc. | Gap | AUC | Iter. |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| CV=5 | Full NN | 50 | `(50, 25)` | `relu` | 0.0001 | 0.8919 | 0.9102 | 0.8904 | 0.0198 | 0.9425 | 43 |
| CV=5 | Reduced NN | 13 | `(25, 10)` | `relu` | 0.001 | 0.8914 | 0.9026 | 0.8935 | 0.0091 | 0.9416 | 39 |
| CV=10 | Full NN | 50 | `(50, 25)` | `relu` | 0.01 | 0.8909 | 0.9137 | 0.8965 | 0.0173 | 0.9360 | 48 |
| CV=10 | Reduced NN | 13 | `(25, 10)` | `relu` | 0.001 | 0.8914 | 0.9026 | 0.8935 | 0.0091 | 0.9416 | 39 |
| CV=10 expanded grid | Full NN | 50 | `(50, 25)` | `tanh` | 0.0001 | 0.8899 | 0.9224 | 0.8828 | 0.0396 | 0.9309 | 69 |
| CV=10 expanded grid | Reduced NN | 13 | `(25, 10)` | `relu` | 0.01 | 0.8955 | 0.9011 | 0.8889 | 0.0122 | 0.9388 | 36 |

Across these runs, test accuracy stayed around 0.88-0.90 and AUC stayed around 0.93-0.94. This suggests that the neural network performance was relatively stable for this dataset. The best test accuracy was 0.8965 from the full neural network with 10-fold CV, while the best AUC was 0.9425 from the full neural network with 5-fold CV. The reduced models were consistently close to the full models despite using only 13 inputs.

The ROC results also show that both the full and reduced neural networks had strong class-separation ability, with AUC values above 0.93. The difference between the full and reduced models was small. This indicates that the decision-tree-selected reduced feature set retained most of the predictive information needed by the neural network.

The expanded hyperparameter search did not consistently improve test performance. In particular, the expanded-grid full neural network had the largest train-test gap, suggesting greater overfitting risk. Therefore, further increasing the search space is unlikely to provide a major improvement without a more systematic validation strategy.

## 4.5 Neural Network Issues

Neural networks can model nonlinear relationships between weather, soil, seed, nursery, and variety variables, but they also introduce several issues. First, they are less interpretable than decision trees or logistic regression because the learned relationships are distributed across hidden-layer weights. This makes it difficult to directly explain why a single prediction was made.

Second, neural networks are sensitive to hyperparameter settings and random initialisation. The results changed slightly when using different CV settings and search ranges, which is why GridSearchCV and early stopping were used. Third, neural networks can overfit when the architecture is too flexible for the dataset. The expanded-grid full model showed this risk with a larger train-test accuracy gap. Finally, neural networks are more computationally expensive than simpler models because they require iterative training and repeated fitting during GridSearchCV.

## 4.6 Summary

The full neural network achieved the highest test accuracy in the main comparison, but the reduced neural network achieved very similar performance using only 13 input variables. Therefore, feature selection was useful for reducing model complexity and improving interpretability, even though it did not clearly outperform the full model on every metric. Overall, the neural network models appear to have reached a stable performance level of about 0.89 accuracy and above 0.93 AUC on this dataset.
