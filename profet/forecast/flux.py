import numpy as np
import pandas as pd
import torch
from profet.dataset.clean import clean_dataset
from profet.utils import read_json_as_dict
from NN_Flujo_Cobranza.neuralnet.nets.model01 import Net01


def predict(raw_data, model, log, time_delta, date_min, date_max):
    """Predict flux with an arbitray PyTorch neural network model.
    
    Args:
        raw_data (pd.DataFrame): The raw dataset with the instances for prediction.
        model: The PyTorch trained model of the neural netowrk.
        log (dict[str, any]): The dictionary with training history of the model.
        time_delta (str): For monthly predictions 'monthly'.
        date_min (dict[str, str]): The lower date considered to make predictions,
            e.g. {'year': '2023', 'month': '01', 'day': '01'}.
        date_max (dict[str, str]): The upper date considered to make predictions,
            e.g. {'year': '2024', 'month': '02', 'day': '28'}.

    Returns:
        pd.DataFrame: The flux prediction beetwen the specified dates in a pd.DataFrame.
    
    """

    # Get training history data from log
    log_features = log['log_features']
    min_max_features = log['min_max_features']
    train_features = log['train_features']
    scalers = log['scalers']

    for feature in min_max_features:
        scalers[feature]['min'] = float(scalers[feature]['min'])
        scalers[feature]['max'] = float(scalers[feature]['max'])

    # Clean the raw data
    data = clean_dataset(raw_data, scalers, log_features, min_max_features)

    # Get inputs for the neural network
    input_features = [x for x in train_features if x != 'dias_atraso']
    inputs = torch.tensor(data[input_features].to_numpy()).float()

    # Make predictions
    predictions = model(inputs)
    transformed_predictions = transform_predictions(predictions, scalers)
    data['pred'] = transformed_predictions

    # Calcular fecha-predecida de pago
    data['fecha_pred'] = data['fecha']
    data['fecha_pred'] += pd.to_timedelta(np.round(data['pred']), unit="D")

    # Parse dates
    data['año_pred'] = data['fecha_pred'].dt.year
    data['mes_pred'] = data['fecha_pred'].dt.month
    data['dia_pred'] = data['fecha_pred'].dt.day
    data['año'] = data['fecha'].dt.year
    data['mes'] = data['fecha'].dt.month
    data['dia'] = data['fecha'].dt.day
    data['año_pag'] = data['fecha_pagada'].dt.year
    data['mes_pag'] = data['fecha_pagada'].dt.month
    data['año_pag'] = data['fecha_pagada'].dt.year

    if time_delta == 'monthly':
        flux_df = monthly_flux(data, date_min, date_max)

    return flux_df


def transform_predictions(predictions, scalers):
    """Pasar la predicción al espacio normal."""

    min_val = scalers['dias_atraso']['min']
    max_val = scalers['dias_atraso']['max']

    # Pasar tensor a numpy array
    with torch.no_grad():
        predictions = predictions.numpy()

    # Quitar escala min-max
    predictions = (max_val - min_val) * predictions + min_val

    # Quitar escala logaritmica
    predictions = np.exp(predictions) - 1

    return predictions


def monthly_flux(data, date_min, date_max):
    """Prediction of the flux with a monthly time discretization."""

    # Parse upper bound of dates
    year_min = date_min['year']
    month_min = date_min['month']
    day_min = date_min['day']
    datetime_min = pd.to_datetime(f"{year_min}-{month_min}-{day_min} 00:00:00+00:00")

    # Parse lower bound of dates
    year_max = date_max['year']
    month_max = date_max['month']
    day_max = date_max['day']
    datetime_max = pd.to_datetime(f"{year_max}-{month_max}-{day_max} 00:00:00+00:00")

    # Neural network prediction
    flux_nn = (
        data[['fecha_pred', 'año_pred', 'mes_pred', 'monto_pago']]
            .query("fecha_pred >= @datetime_min")
            .query("fecha_pred <= @datetime_max")
            .groupby(['año_pred', 'mes_pred'])['monto_pago']
            .agg("sum")
            .reset_index()
    )

    # Prediction based on contract date
    flux_dev = (
        data[['fecha', 'año', 'mes', 'monto_pago']]
            .query("fecha >= @datetime_min")
            .query("fecha <= @datetime_max")
            .groupby(['año', 'mes'])['monto_pago']
            .agg("sum")
            .reset_index()
    )

    # Real flux
    flux_real = (
        data[['fecha_pagada', 'año_pag', 'mes_pag', 'monto_pago', 'pagada']]
            .query("pagada == True")
            .query("fecha_pagada >= @datetime_min")
            .query("fecha_pagada <= @datetime_max")
            .groupby(['año_pag', 'mes_pag'])['monto_pago']
            .agg("sum")
            .reset_index()
    )

    # Pass units to mdp
    flux_nn['monto_pago'] /=  1e6
    flux_dev['monto_pago'] /=  1e6
    flux_real['monto_pago'] /= 1e6

    # Join all predictions together
    flux = pd.DataFrame({
        "year": flux_nn['año_pred'],
        "month": flux_nn['mes_pred'],
        "nn": flux_nn['monto_pago'],
        "dev": flux_dev['monto_pago'],
        "real": flux_real['monto_pago']
    })

    flux['date'] = (pd.to_datetime({
        'year': flux['year'],
        'month': flux['month'],
        'day': 1
    }))

    return flux


def test_monthly_flux():
    """Test the monthly flux predictions."""

    # Load raw data for the test
    raw_data = pd.read_parquet("data/v1/dataset_v1.parquet")

    # Read training history
    log_path = "NN_Flujo_Cobranza/models/model01/lr_gridsearch/log_lr_0.json"
    log = read_json_as_dict(log_path)

    # Load an instance of the model and its trained state
    model = Net01()
    device = torch.device('cpu')
    model_state_path = f"NN_Flujo_Cobranza/{log['model_state']}"
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    model.eval()

    flux_df = predict(
        raw_data=raw_data,
        model=model,
        log=log,
        time_delta='monthly',
        date_min={'year': '2023', 'month': '01', 'day': '01'},
        date_max={'year': '2024', 'month': '04', 'day': '31'}
    )

    print(flux_df)


if __name__ == "__main__":

    test_monthly_flux()
