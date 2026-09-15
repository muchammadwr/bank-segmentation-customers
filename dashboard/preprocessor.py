import pandas as pd


def create_age_group(age):
    if 18 <= age <= 32:
        return "rendah"
    elif 33 <= age <= 55:
        return "sedang"
    elif 56 <= age <= 80:
        return "tinggi"

    raise ValueError("CustomerAge harus berada antara 18 dan 80.")


def preprocess_clustering(data, bundle):
    prepared_data = data.loc[
        :,
        bundle["feature_columns"],
    ].copy()

    for column in bundle["categorical_columns"]:
        encoder = bundle["encoders"][column]

        prepared_data[column] = encoder.transform(prepared_data[column].astype(str))

    return prepared_data
