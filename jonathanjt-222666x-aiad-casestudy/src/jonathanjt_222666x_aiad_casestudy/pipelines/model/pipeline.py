from kedro.pipeline import Pipeline, node
from .nodes import (
    split_data,
    rbf_SVM,
    Logistic_Regression,
    Decision_Tree,
    Random_Forest,
    Gradient_Boosting,
    Neural_Network,
    cross_validate_models,
    tune_hyperparameters,
    tune_random_forest,
    train_voting_classifier
)

def pipeline_model() -> Pipeline:
    return Pipeline(
        [
            # Data Splitting
            node(
                func=split_data,
                inputs="training_data",
                outputs=["train_X", "train_Y", "test_X", "test_Y"],
                name="split_data_node",
            ),
            # Individual Models
            node(
                func=rbf_SVM,
                inputs=["train_X", "train_Y", "test_X", "test_Y"],
                outputs=["svm_rbf_accuracy", "svm_linear_accuracy"],
                name="train_rbf_svm_node",
            ),
            node(
                func=Logistic_Regression,
                inputs=["train_X", "train_Y", "test_X", "test_Y"],
                outputs="logistic_regression_accuracy",
                name="train_logistic_regression_node",
            ),
            node(
                func=Decision_Tree,
                inputs=["train_X", "train_Y", "test_X", "test_Y"],
                outputs="decision_tree_accuracy",
                name="train_decision_tree_node",
            ),
            node(
                func=Random_Forest,
                inputs=["train_X", "train_Y", "test_X", "test_Y"],
                outputs="random_forest_accuracy",
                name="train_random_forest_node",
            ),
            node(
                func=Gradient_Boosting,
                inputs=["train_X", "train_Y", "test_X", "test_Y"],
                outputs="gradient_boosting_accuracy",
                name="train_gradient_boosting_node",
            ),
            node(
                func=Neural_Network,
                inputs=["train_X", "train_Y", "test_X", "test_Y"],
                outputs="neural_network_accuracy",
                name="train_neural_network_node",
            ),
            # Cross-Validation
            node(
                func=cross_validate_models,
                inputs="training_data",
                outputs="cross_validation_results",
                name="cross_validate_models_node",
            ),
            # Hyperparameter Tuning
            node(
                func=tune_hyperparameters,
                inputs="training_data",
                outputs=["best_score_hyperparameters", "best_estimator_hyperparameters"],
                name="tune_hyperparameters_node",
            ),
            node(
                func=tune_random_forest,
                inputs="training_data",
                outputs=["best_score_random_forest", "best_estimator_random_forest"],
                name="tune_random_forest_node",
            ),
            # Ensemble Models
            node(
                func=train_voting_classifier,
                inputs=["train_X", "train_Y", "test_X", "test_Y", "training_data"],
                outputs=["voting_classifier_accuracy", "voting_classifier_cross_val_accuracy"],
                name="train_voting_classifier_node",
            )
        ]
    )
