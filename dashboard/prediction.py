from preprocess import (
    add_age_group,
    preprocess_clustering,
    preprocess_classification,
)


def predict_cluster(
    data,
    model,
    encoders,
    scalers,
    age_config,
):
    input_data = add_age_group(
        data,
        scalers["CustomerAge"],
        age_config,
    )

    processed_data = preprocess_clustering(
        input_data,
        encoders,
        scalers,
    )

    predictions = model.predict(processed_data)

    result = input_data.copy()
    result["Cluster"] = predictions

    return result


def predict_classification(
    data,
    model,
    schema,
    scalers,
    age_config,
):
    input_data = add_age_group(
        data,
        scalers["CustomerAge"],
        age_config,
    )

    processed_data = preprocess_classification(
        input_data,
        schema,
    )

    predictions = model.predict(processed_data)

    result = input_data.copy()
    result["Target"] = predictions

    return result