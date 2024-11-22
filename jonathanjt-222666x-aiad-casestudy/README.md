# Titanic Survival Prediction Project

## Personal Information

- Full Name: Jonathan James Thomas
- Email address: 222666X@mymail.nyp.edu.sg

## Overview of the Submitted Folder and Folder Structure
/titanic-survival-prediction
│
├── /data                     # Data directory, contains raw data, models, and test results
│   ├── /01_original_data     # Raw data
│   ├── /02_processed_data    # Cleaned and Processed Data
│   └── /03_model             # Model Folder
│   └── /04_prediction_data   # Prediction Results
│ 
├── /conf                     # Configuration directory, contains Kedro parameter settings, pipeline configurations, etc.
│   ├── /base                 # Base configurations
│       ├── catalog.yml       # Dataset configuration
│       └── parameters.yml    # Parameter configuration
│ 
├── /src                      # Python scripts for data preparation, model training, and prediction
│   ├── /data_prep            # Data preparation pipeline
│   ├── /model                # Model training pipeline
│   └── /prediction           # Prediction pipeline
│ 
│── /current_model.pkl        # Final model for submission
│── /eda.ipynb                # Exploratory Data Analysis notebook
│── /eda.pdf                  # Exploratory Data Analysis in PDF
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies file
└── pyproject.toml            # Build Specifications


---

## Programming Language

**Programming Language**  
Python: Version 3.10

**Runtime Environment Prerequisites**  
Operating System:
  - MacOS: 

## List of Libraries Used
ipython>=8.10
jupyterlab>=3.0
kedro~=0.19.9
kedro-datasets[pandas-csvdataset, pandas-exceldataset, pandas-parquetdataset]>=3.0
kedro-viz>=6.7.0
kedro[jupyter]
notebook
scikit-learn~=1.5.1

## Key Findings from the EDA and Feature Engineering

### Key Findings:

- **Age**:  
  A significant relationship exists between age and survival rate. Younger passengers had higher survival chances, with a slight variation depending on gender and social class.

- **Fare/Class**:  
  A positive correlation was observed between fare and survival probability. Passengers who paid higher fares generally had better survival outcomes, linked to their social class and access to resources.

- **Gender**:  
  Gender was a critical factor, with female passengers having significantly higher survival rates than males, reflecting historical lifeboat allocation and rescue priorities.

- **Family Size**:  
  Passengers traveling with family members had a higher likelihood of survival compared to those traveling alone, suggesting the importance of familial support during emergencies.

---

## Feature Engineering Choices:

### Handling Missing Age Values:
- **Median Imputation**:  
  Missing values for age were filled using the median value, as it is less sensitive to outliers.

### Standardization of Fare:
- Fare values were standardized using Min-Max Scaling or Z-score Normalization to ensure equal influence during model training and improve convergence speed.

### Family Size:
- Combined SibSp (siblings/spouses) and Parch (parents/children) to create a new feature `Family_Size`, and introduced a binary feature `Alone` indicating if a passenger was traveling alone (Family_Size = 0).

### Improved Age Segmentation:
- Created an `Age_band` feature by segmenting age into the following categories:
  - 0–5 years: Infant/Toddler
  - 6–16 years: Child
  - 17–32 years: Young Adult
  - 33–48 years: Adult
  - 49–65 years: Middle-aged
  - 65+ years: Senior.

### Embarked Feature:
- Missing values in the Embarked column were imputed using contextual characteristics like gender and fare, instead of using the most frequent value ('S').
---

## Instructions for Executing the Pipeline and Modifying Any Parameters

### 1. Execute the Entire Workflow
To run the entire pipeline sequence (data preprocessing, model training, and prediction), use the following command:

```bash
kedro run
```

### 2. Execute Individual Pipelines
If you want to focus on specific parts of the workflow, you can run pipelines individually using the --pipeline flag.

- a. Data Processing Pipeline (data_prep): Prepares the raw dataset for model training by handling missing values, encoding features, and scaling data.

```bash
kedro run --pipeline data_prep  
```
- b. Model Training Pipeline (model):Trains and fine-tunes machine learning models using the preprocessed dataset.

```bash
    kedro run --pipeline model  
```
- c. Prediction Pipeline (prediction):Generates predictions and evaluates the trained model's performance on the test dataset.
```bash
    kedro run --pipeline prediction 
``` 

###  3.Feature Selection for Model Training
- Pclass Survived,Pclass,Age,SibSp,Parch,Fare,HasCabin,FamilySize,IsAlone,FareBin,Age_band,Sex_male,Embarked_Q,Embarked_S,Title_Miss,Title_Mr,Title_Mrs,Title_Other
- Sex 
- SibSp 
- Parch 
- Embarked 
- Initial 
- Age_band 
- Family_Size 
- Alone 
- Survived_Prediction

###  4. Modifying Parameters
Pipeline parameters can be adjusted to fine-tune the workflow. These parameters are located in the /conf directory under parameters.yml.

    data_split:
      test_size: 0.3
      random_state: 42

## Below is a flow chart of the pipeline
![App Screenshot](https://github.com/2-1GPAproject/eda_224531m/blob/main/flow_chart.png?raw=true)

## Explanation of models choose
### 1. LGBMClassifier (LightGBM)

- Efficiency: LightGBM speeds up the training process by using leaf-wise tree growth, which can be more efficient than traditional level-wise methods, especially on small datasets. This results in faster training times while maintaining high accuracy.

- High Accuracy: The tree-building algorithm in LightGBM is optimized for better predictive accuracy, allowing it to capture patterns in the data even when the dataset size is small.

- Overfitting Control: LightGBM has strong regularization techniques such as num_leaves, max_depth, and min_data_in_leaf to prevent overfitting, which is critical when working with small datasets where overfitting can be a major concern.

### 2. Random Forest

- Robustness: Random Forest is resistant to noise and overfitting by averaging the results of multiple decision trees trained on random subsets of the data. This means the model is less likely to be impacted by small variations or outliers in the data.

- Overfitting Reduction: The model trains on various subsets of the data and uses random feature selection, which reduces the likelihood of overfitting, especially important when working with small datasets where overfitting is a common issue.

- Versatility: Random Forest can handle a mix of numeric and categorical features and deal with missing values naturally, making it adaptable to various types of data, which is an advantage in many real-world scenarios.

### 3. AdaBoost (Adaptive Boosting)

- Enhances Weak Learners: AdaBoost focuses on the instances that were misclassified by previous weak learners by adjusting the weights of these samples. This iterative process improves the overall model accuracy, especially with small datasets.

-  Adaptability to Small Datasets: AdaBoost reweights the training samples, which helps the model focus on harder-to-classify examples, making it more effective for small datasets, particularly when the dataset has noisy or outlier data.

-  Effective with Imbalanced Data: AdaBoost tends to perform well on imbalanced datasets because it gives more weight to the misclassified instances, often improving the model’s ability to classify the minority class.

![App Screenshot](https://raw.githubusercontent.com/2-1GPAproject/eda_224531m/refs/heads/main/KFold.png)


- LGBMClassifier: Achieved a high cross-validation mean score (0.8171) with a relatively low standard deviation (0.0430), indicating both strong performance and stability. This makes it a robust choice for the prediction task.

- Random Forest: Similarly, it showed stable performance with a CV mean of 0.8115 and a standard deviation of 0.0448, which further validates its suitability for the task.



![App Screenshot](https://raw.githubusercontent.com/2-1GPAproject/eda_224531m/refs/heads/main/box.png)

Boxplot Distribution:

The accuracy distributions for Random Forest and LGBMClassifier are fairly concentrated, indicating strong model stability.
Both models’ accuracy scores are close to or above the upper quartile of other models, performing better than most other algorithms.

![App Screenshot](https://raw.githubusercontent.com/2-1GPAproject/eda_224531m/refs/heads/main/maxtris.png)

Confusion Matrix Performance:

Random Forest and LGBMClassifier show high accuracy in predicting the positive class 1, with a higher rate of True Positives.
Both models have fewer False Positives and False Negatives, demonstrating strong predictive power.

SVM:SVM has a relatively slow training speed, especially with the rbf kernel method. It significantly increases computational complexity when handling large datasets.

AdaBoost: Although AdaBoost was not explicitly included in the CV results, it can still be an excellent option, particularly in scenarios where boosting weak learners could lead to improved performance, especially with noisy or imbalanced datasets.


## 4. Model Hyperparameter Optimization

LGBMClassifier with RandomizedSearchCV: In order to fine-tune the hyperparameters of the LGBMClassifier, RandomizedSearchCV was used. This technique performs a random search across the hyperparameter space, making it computationally efficient while still providing good results. Given that LGBMClassifier has many hyperparameters (such as num_leaves, max_depth, learning_rate, etc.), RandomizedSearchCV helps in efficiently finding the optimal configuration without exhaustive search.

Random Forest and AdaBoost with GridSearchCV: For Random Forest and AdaBoost, GridSearchCV was used to exhaustively search through a grid of hyperparameter values. Both models have fewer hyperparameters compared to LGBMClassifier, so a full search over all possible parameter combinations is more computationally feasible and effective for improving the models' performance.


# Evaluation of the models

LGBMClassifier (LightGBM)
Strong performance for the negative class (high recall), but struggles with the positive class (lower recall).
```bash
Classification Report:
               precision    recall  f1-score   support

           0       0.82      0.91      0.86       165
           1       0.82      0.68      0.74       103

    accuracy                           0.82       268
   macro avg       0.82      0.79      0.80       268
weighted avg       0.82      0.82      0.82       26
``` 

Random Forest
Good performance for the negative class, but a relatively low recall for the positive class.
```bash
Classification Report:
               precision    recall  f1-score   support

           0       0.82      0.92      0.87       165
           1       0.84      0.67      0.75       103

    accuracy                           0.82       268
   macro avg       0.83      0.80      0.81       268
weighted avg       0.83      0.82      0.82       268
```
AdaBoost (Adaptive Boosting)
Performs well overall, especially in handling class imbalance, with good precision and recall for both classes, and slightly better overall accuracy than the other two models.
```bash
Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.92      0.87       165
           1       0.84      0.71      0.77       103

    accuracy                           0.84       268
   macro avg       0.84      0.81      0.82       268
weighted avg       0.84      0.84      0.83       268
```
ROC analysis
![App Screenshot](https://raw.githubusercontent.com/2-1GPAproject/eda_224531m/refs/heads/main/roc.png)

LightGBM:
- The AUC value is 0.87, indicating that the LightGBM model has strong ability to distinguish between positive and negative classes.
- The curve performs well in the top-left area, suggesting that the model has a low tolerance for misclassifications, making it suitable for high-precision scenarios.

Random Forest:
- The AUC value is also 0.87, matching LightGBM.
- The shape of the curve is similar to LightGBM, indicating good classification ability, but it may be slightly inferior under certain probability thresholds.

AdaBoost:
- The AUC value is 0.86, slightly lower than LightGBM and Random Forest.
- The ROC curve is slightly away from the top-left corner, suggesting that the model may be more sensitive to noise or sample weights.

Conclusion:
- The classification abilities of the three models are close, with LightGBM and Random Forest slightly outperforming AdaBoost in terms of ROC performance.
- If high discrimination ability is required, LightGBM and Random Forest are more suitable.


Learning curve analysis
![App Screenshot](https://raw.githubusercontent.com/2-1GPAproject/eda_224531m/refs/heads/main/learning%20curves.png)


LightGBM:

- The training score is significantly higher than the cross-validation score, indicating that the model has some degree of overfitting.
- As the dataset size increases, the cross-validation score gradually improves, suggesting an improvement in the model's generalization performance.
- Overall, LightGBM is sensitive to data size and may require further tuning to mitigate overfitting.

Random Forest:
- The training score is high and stable, indicating that the model fits the training data well but has weak generalization ability.
- As the dataset size increases, the cross-validation score grows slowly, and the gap remains significant.
- Random Forest tends to overfit when handling small datasets and is suitable for scenarios where there are complex interactions between features.

AdaBoost:

- In the initial stages, the training score is high, but as the dataset size increases, the training and cross-validation scores gradually converge.
- The learning curve for AdaBoost shows strong generalization ability, making it suitable for medium and small datasets with low sensitivity to overfitting.

Conclusion:
- AdaBoost shows the most stable performance on the learning curve, with the best generalization ability.
- To reduce overfitting, further optimization of LightGBM and Random Forest could be considered, such as through regularization or feature selection.

