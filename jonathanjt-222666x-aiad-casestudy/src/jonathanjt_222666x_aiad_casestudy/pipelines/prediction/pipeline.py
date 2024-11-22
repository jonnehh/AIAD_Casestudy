from kedro.pipeline import Pipeline, node
from .nodes import (process_miss_data_test, 
                    feature_engineering_test,
                    drop_unnecessary_features_test,
                    predictions)

def pipeline_prediction() -> Pipeline:
    return Pipeline(
        [
            # Node to handle missing data
            node(
                func=process_miss_data_test, 
                inputs="titanic_data_test",  # Input dataset from catalog (titanic_data_train)
                outputs="processed_missing_data_test",  # Output dataset to be passed to next node
                name="process_missing_data_test",
            ),
            # Node to perform feature engineering
            node(
                func=feature_engineering_test, 
                inputs="processed_missing_data_test",  # Use the output from previous node
                outputs="engineered_features_test",  # Output dataset to be passed to next node
                name="feature_engineering_test",
            ),
            # Node to drop unnecessary features
            node(
                func=drop_unnecessary_features_test, 
                inputs="engineered_features_test",  # Use the output from previous node
                outputs="test_data",  # Output dataset to be saved as final data
                name="drop_unnecessary_features_test",
            ),
            node(
                func=predictions,
                inputs=["test_data","titanic_data_test"],
                outputs="prediction_data",
                name="predictons",
            ),

        ]
    )
