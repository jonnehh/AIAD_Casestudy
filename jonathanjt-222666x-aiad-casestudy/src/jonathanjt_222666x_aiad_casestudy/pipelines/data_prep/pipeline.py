from kedro.pipeline import Pipeline, node
from .nodes import process_miss_data, feature_engineering, drop_unnecessary_features

def pipeline_data_prep() -> Pipeline:
    return Pipeline(
        [
            # Node to handle missing data
            node(
                func=process_miss_data, 
                inputs="titanic_data_train",  # Input dataset from catalog (titanic_data_train)
                outputs="processed_missing_data",  # Output dataset to be passed to next node
                name="process_missing_data_node",
            ),
            # Node to perform feature engineering
            node(
                func=feature_engineering, 
                inputs="processed_missing_data",  # Use the output from previous node
                outputs="engineered_features",  # Output dataset to be passed to next node
                name="feature_engineering_node",
            ),
            # Node to drop unnecessary features
            node(
                func=drop_unnecessary_features, 
                inputs="engineered_features",  # Use the output from previous node
                outputs="training_data",  # Output dataset to be saved as final data
                name="drop_unnecessary_features_node",
            ),
        ]
    )
