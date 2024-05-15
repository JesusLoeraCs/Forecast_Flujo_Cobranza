"""Flux predictions with model01."""

import torch
from profet.dataset.extract import dataset_from_mongo
from profet.utils import read_json_as_dict, load_to_mongo
from profet.forecast.flux import predict
from NN_Flujo_Cobranza.neuralnet.nets.model01 import Net01


DATASET_DB_NAME = 'dataset_flujo_cobranza'
DATASET_COLLECTION_NAME = 'v1'

URI = "mongodb://localhost:27017/"
LOWER_DATE = {'year': '2023', 'month': '01', 'day': '01'}
UPPER_DATE = {'year': '2024', 'month': '06', 'day': '30'}

FLUX_DB_NAME = 'forecast_flujo_cobranza'
MODEL_NAME = 'model01'

def monthly_flux_model01():
    """Predict the monthly flux with model01 and load it into MongoDB."""

    print("\nReading dataset from MongoDB...")

    # Read data from MongoDB
    raw_data = dataset_from_mongo(
        database_name=DATASET_DB_NAME,
        collection_name=DATASET_COLLECTION_NAME,
        uri=URI,
        show_progress_bar=True
    )

    print("Loading neural network...")

    # Read training history
    log_path = "NN_Flujo_Cobranza/models/model01/lr_gridsearch/log_lr_0.json"
    log = read_json_as_dict(log_path)

    # Load an instance of the model and its trained state
    model = Net01()
    device = torch.device('cpu')
    model_state_path = f"NN_Flujo_Cobranza/{log['model_state']}"
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    model.eval()

    print("\nCleaning data and making predictions...")

    monthly_flux, amort_flux = predict(
        raw_data=raw_data,
        model=model,
        log=log,
        time_delta='monthly',
        date_min=LOWER_DATE,
        date_max=UPPER_DATE
    )

    print("\nLoading monthly predictions to MongoDB...")

    load_to_mongo(
        monthly_flux,
        mongo_uri=URI,
        database_name=FLUX_DB_NAME,
        collection_name=f'{MODEL_NAME}_mensual'
    )

    print("\nLoading granular predictions to MongoDB...")

    load_to_mongo(
        amort_flux,
        mongo_uri=URI,
        database_name=FLUX_DB_NAME,
        collection_name=f'{MODEL_NAME}_amortizacion'
    )

    print("Process finished.\n")
