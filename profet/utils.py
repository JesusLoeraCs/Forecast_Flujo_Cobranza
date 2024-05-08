import json
import pandas as pd
from pymongo import MongoClient


def read_json_as_dict(json_path):

    with open(json_path, 'r', encoding='utf-8') as json_file:

        data = json.load(json_file)

        # Case 1: json is load directly into a dictionary
        if isinstance(data, dict):
            dictionary = data

        # Case2: json is load into a string
        elif isinstance(data, str):
            data = json.loads(data)
            if isinstance(data, dict):
                dictionary = data

    return dictionary


def clean_for_mongo(df: pd.DataFrame) -> pd.DataFrame:
    """Preparar el dataframe para subirse a MongoDB"""

    df = df.copy()

    # Convertir fechas al formato formato de fecha y hora ISO 8601
    for col in df.columns:
        # Convertir fechas al formato formato de fecha y hora ISO 8601
        if df[col].dtype in ['datetime64[ns, UTC]', 'datetime64[us, UTC]']:
            df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    return df


def load_to_mongo(
        data: pd.DataFrame,
        uri: str, db_name: str,
        collection_name: str
    ) -> None:
    """Guardar dataset.
    
    Args:
        dataset (pd.DataFrame): Conjunto de datos de amortizaciones.
        destino (str): Destino de carga del datset ('csv', 'parquet' o 'mongo').
    
    """

    dataset_mongo = clean_for_mongo(data)

    # Establecer conexión con MongoDB
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    collection.delete_many({})

    # Convertir el DataFrame a formato JSON y cargarlo en la colección
    records = dataset_mongo.to_dict(orient='records')
    collection.insert_many(records)

    # Cerrar la conexión
    client.close()
