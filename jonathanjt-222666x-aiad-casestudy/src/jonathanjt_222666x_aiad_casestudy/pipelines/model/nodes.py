from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import metrics
import warnings
import os
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Data Splitting
def split_data(train_data, test_size=0.2, random_state=888):
    """
    Split the dataset into training and testing sets.
    """
    train, test = train_test_split(
        train_data,
        test_size=test_size,
        random_state=random_state,
        stratify=train_data["Survived"],
    )
    train_X = train.drop(columns=['Survived'])  # Features
    train_Y = train['Survived']  # Labels
    test_X = test.drop(columns=['Survived'])
    test_Y = test['Survived']
    return train_X, train_Y, test_X, test_Y


# Individual Models
def rbf_SVM(train_X, train_Y, test_X, test_Y):
    print("Training Radial SVM...")
    model_rbf = svm.SVC(kernel="rbf", C=1, gamma=0.1)
    model_rbf.fit(train_X, train_Y)
    prediction_rbf = model_rbf.predict(test_X)
    accuracy_rbf = metrics.accuracy_score(prediction_rbf, test_Y)
    print(f"Radial SVM Accuracy: {accuracy_rbf:.4f}")

    print("Training Linear SVM...")
    model_linear = svm.SVC(kernel="linear", C=0.1, gamma=0.1)
    model_linear.fit(train_X, train_Y)
    prediction_linear = model_linear.predict(test_X)
    accuracy_linear = metrics.accuracy_score(prediction_linear, test_Y)
    print(f"Linear SVM Accuracy: {accuracy_linear:.4f}")

    return accuracy_rbf, accuracy_linear


def Logistic_Regression(train_X, train_Y, test_X, test_Y):
    print("Training Logistic Regression...")
    model = LogisticRegression()
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction, test_Y)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    return accuracy


def Decision_Tree(train_X, train_Y, test_X, test_Y):
    print("Training Decision Tree...")
    model = DecisionTreeClassifier()
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction, test_Y)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    return accuracy


def Random_Forest(train_X, train_Y, test_X, test_Y):
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction, test_Y)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    return accuracy


def Gradient_Boosting(train_X, train_Y, test_X, test_Y):
    print("Training Gradient Boosting...")
    model = GradientBoostingClassifier(n_estimators=500, random_state=888, learning_rate=0.1)
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction, test_Y)
    print(f"Gradient Boosting Accuracy: {accuracy:.4f}")
    return accuracy


def Neural_Network(train_X, train_Y, test_X, test_Y):
    print("Training Neural Network...")
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=888)
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction, test_Y)
    print(f"Neural Network Accuracy: {accuracy:.4f}")
    return accuracy


# Cross-Validation
def cross_validate_models(data):
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    classifiers = [
        "Linear Svm",
        "Radial Svm",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Neural Network",
    ]
    models = [
        svm.SVC(kernel="linear"),
        svm.SVC(kernel="rbf"),
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier(n_estimators=500, random_state=888, learning_rate=0.1),
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=888),
    ]

    kfold = KFold(n_splits=10, random_state=888, shuffle=True)
    print("Cross-validation results:")
    results = {}
    for classifier, model in zip(classifiers, models):
        scores = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
        mean_score = scores.mean()
        results[classifier] = mean_score
        print(f"{classifier} Cross-Validation Accuracy: {mean_score:.4f}")
    return results


# Hyperparameter Tuning
def tune_hyperparameters(data):
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    hyperparameters = {"C": [0.1, 0.5, 1], "gamma": [0.1, 0.5, 1], "kernel": ["rbf", "linear"]}
    grid_search = GridSearchCV(
        estimator=svm.SVC(), param_grid=hyperparameters, scoring="accuracy"
    )
    grid_search.fit(X, Y)
    print(f"Best Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_score_, grid_search.best_params_


def tune_random_forest(data):
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    hyperparameters = {"n_estimators": [100, 200, 300]}
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=0),
        param_grid=hyperparameters,
        scoring="accuracy",
    )
    grid_search.fit(X, Y)
    print(f"Best Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_score_, grid_search.best_params_


# Ensemble Models
def train_voting_classifier(train_X, train_Y, test_X, test_Y, data):
    print("Training Voting Classifier...")
    ensemble_model = VotingClassifier(
        estimators=[
            ("SVM", svm.SVC(probability=True, kernel="rbf", C=0.5, gamma=0.1)),
            ("Logistic", LogisticRegression(C=0.05)),
            ("RandomForest", RandomForestClassifier(n_estimators=100)),
            ("GradientBoosting", GradientBoostingClassifier(n_estimators=500)),
        ],
        voting="soft",
    )
    ensemble_model.fit(train_X, train_Y)
    accuracy = ensemble_model.score(test_X, test_Y)
    print(f"Ensemble Model Accuracy: {accuracy:.4f}")
    cross_val_mean = cross_val_score(
        ensemble_model, data.iloc[:, 1:], data.iloc[:, 0], cv=10, scoring="accuracy"
    ).mean()
    print(f"Ensemble Cross-Validation Mean Accuracy: {cross_val_mean:.4f}")

    # For Model Saving
    model_dir = "data/03_model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "current_model.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(ensemble_model, file)
        print("model saved")
    #
    return accuracy, cross_val_mean