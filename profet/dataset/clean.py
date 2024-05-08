"""Process data for make predictions with neural networks."""

import pandas as pd
import numpy as np
from profet.utils import read_json_as_dict


def clean_dataset(
        raw_data: pd.DataFrame,
        scalers: dict[dict[str, float]],
        log_features: list[str],
        min_max_features: list[str]
    ) -> pd.DataFrame:
    """Prepare dataset for neural networks.
    
    Args:
        raw_data (pd.DataFrame): The amortizations dataset that will be clean.
        train_features (list[str]): The list of inputs + target features for training.
        log_features (list[str]): The list of features that are transformed to
            logarithmic scale.
        min_max_features (list[str]): The list of features that are transformed to
            min-max scale.
    
    """

    data = raw_data.copy()

    # Convert dtypes
    data['fecha'] = pd.to_datetime(data['fecha'])
    data['fecha_pagada'] = pd.to_datetime(data['fecha_pagada'])
    data = data.astype({'empresa_id': float, 'monto_pago': float})

    # Make predictions in data after 2023
    x_min = pd.to_datetime("2023-01-01 00:00:00+00:00")
    data = (
        data.query("tipo_descuento == 'retencion'")
            .query("fecha >= @x_min")
    )

    # Fill missing values
    data = impute_data(data)

    # Map categorical variables
    data['frecuencia_cobro'] = (
        data['frecuencia_cobro'].map({'semanal': 7.0, 'decenal': 7.0, 'catorcenal': 14.0, 
                                      'quincenal': 15.0, 'mensual': 30})
    )

    # Apply constraints to data
    data = bound_data(data)

    # Apply logaritmic scale and training min-max scale
    data = scale_log(data, features=log_features)
    data = scale_min_max(data, scalers=scalers, features=min_max_features)

    return data


def impute_data(data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing data.
    
    Args:
        data (pd.DataFrame): The dataset with missing values to impute.

    Returns:
        pd.DataFrame: The dataset with no missing values.
    
    """

    # If dias_atraso is NaN then the credit is paid up to date
    data.fillna({'dias_ultimo_atraso': 0.0}, inplace=True)

    # Statistical measures that cannot be calculated due to lack of history
    data.fillna({'media_dac': 0.0, 'std_dac': 0.0, 'moda_dac': 0.0}, inplace=True)
    data.fillna({'media_dae': 0.0, 'std_dae': 0.0, 'moda_dae': 0.0}, inplace=True)

    # Replace NaN en credito_4m100 with False.
    data = data.fillna({'credito_4m100': False})

    # Possible missing empresa_id to new company
    min_value = data['empresa_id'].min()
    max_value = data['empresa_id'].max()
    missing_values = data['empresa_id'].isna()
    data.loc[missing_values, 'empresa_id'] = np.random.randint(
        low=min_value, high=max_value, size=missing_values.sum()
    )

    # Possible missing frecuencia_cobro imputed with 'catorcenal'
    data.fillna({'frecuencia_cobro': 'catorcenal'}, inplace=True)


    return data


def bound_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply dataset's constraints for deep learning algorithms.
    
    Args:
        data (pd.DataFrame): The dataset that will be transformed.
    
    Returns:
        pd.DataFrame: Dataset with the constraints applied.

    """

    # Negative dias_ultimo_atraso are unpredictable and are converted to on-time payments
    data['dias_ultimo_atraso'] = data['dias_ultimo_atraso'].clip(lower=0)


    return data


def scale_log(
        data: pd.DataFrame,
        features: list[str],
        inverse: bool = False
    ) -> pd.DataFrame:
    """Apply a logaritmic or exponential scale to some features.
    
    Args:
        data (pd.DataFrame): The dataset that will be transformed.
        features (list[str]): The list with the features to be scaled.
        inverse (bool, optional): True if logaritmic scale, false if exponential scale.
    
    Returns:
        pd.DataFrame: Dataset with its specified features scaled.

    """

    for feature in [x for x in features if x != 'dias_atraso']:

        if not inverse:
            data[feature] = np.log(data[feature] + 1)
        else:
            data[feature] = np.exp(data[feature]) - 1

    return data


def scale_min_max(
        data: pd.DataFrame,
        scalers: dict[dict[str, float]],
        features: list[str],
        inverse: bool = False
    ) -> pd.DataFrame:
    """Apply min-max or inverse min-max transform to some features.
    
    Args:
        data (pd.DataFrame): The dataset that will be transformed.
        scalers (dict[dict[str, float]]): The dictionary with the parameters that defines
            the min-max scaler of each feture, e.g {'moda_dae': {'min': 2, 'max': 24}}.
        inverse (bool, optional): True if min-max scale, false if inverse min-max scale.

    Returns:
        pd.DataFrame: Dataset with its specified features scaled.
    
    """

    for feature in features:

        min_val = int(scalers[feature]['min'])
        max_val = int(scalers[feature]['max'])

        if not inverse:
            data[feature] = (data[feature] - min_val) / (max_val - min_val)
        else:
            data[feature] = (max_val - min_val) * data[feature] + min_val

    return data


def test_clean_dataset():
    """Test the function 'clean_dataset'."""

    raw_data = pd.read_parquet("data/v1/dataset_v1.parquet")
    log_path = 'NN_Flujo_Cobranza/models/model01/lr_gridsearch/log_lr_0.json'

    log = read_json_as_dict(log_path)
    scalers = log['scalers']

    for feature in scalers.keys():
        scalers[feature]['min'] = float(scalers[feature]['min'])
        scalers[feature]['max'] = float(scalers[feature]['max'])

    log_features = log['log_features']

    min_max_features = log['min_max_features']

    clean_dataset(
        raw_data=raw_data,
        scalers=scalers,
        log_features=log_features,
        min_max_features=min_max_features
    )


if __name__ == '__main__':

    test_clean_dataset()
