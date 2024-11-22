from kedro.pipeline import Pipeline
from jonathanjt_222666x_aiad_casestudy.pipelines.data_prep import pipeline_data_prep
from jonathanjt_222666x_aiad_casestudy.pipelines.model import pipeline_model
from jonathanjt_222666x_aiad_casestudy.pipelines.prediction import pipeline_prediction

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": pipeline_data_prep() +pipeline_model() +pipeline_prediction(),
        "data_prep": pipeline_data_prep(),
        "model": pipeline_model(),
        'prediction':pipeline_prediction(),
    }
