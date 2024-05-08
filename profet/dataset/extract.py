"""Module for retrieving data for training neural networks."""

import os

import requests
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm


load_dotenv()


def dataset_from_mongo(
        database_name: str,
        collection_name: str,
        uri: str = "mongodb://localhost:27017/",
        batch_size = 1000, 
        show_progress_bar: bool = True
    ) -> pd.DataFrame:
    """Establish a connection to MongoDB to query data collections.
    
    Args:
        database_name (str): The name of the database.
        collection_name (str): The name of the database's collection.
        uri (str, optional): The URI that identifies the server where the databases is
            located.
        show_progress_bar (bool, optional): True to display a bar of execution progress.

    Returns:
        pd.DataFrame: Data collection loaded into a pd.DataFrame.
    
    """

    # Connect to MongoDB
    client = MongoClient(uri)
    db = client[database_name]
    collection = db[collection_name]
    cursor = collection.find()

    # Load data into DataFrame with progress bar
    projection = {"_id": 0}
    cursor = collection.find(projection=projection, batch_size=batch_size)

    if show_progress_bar:
        df = pd.DataFrame(tqdm(cursor, desc="Reading data..."))
    else:
        df = pd.DataFrame(cursor)

    client.close()

    return df


def github_read_parquet(
        file_name: str = os.getenv('FILE_NAME'),
        file_path: str = os.getenv('FILE_PATH'),
        username: str = os.getenv('USERNAME'),
        repository_name: str = os.getenv('REPOSITORY'),
        token: str = os.getenv('TOKEN')
    ) -> pd.DataFrame:
    """Pull data at parquet files for train neural networks from GitHub.
    
    Args:
        file_name (str): The name of the Parquet file.
        file_path (str): The path to the Parquet file in the repository.
        username (str): The username of the repository owner.
        repository_name (str): The name of the repository where the file is located.
        token (str): Token to access a private repository.
    
    Returns:
        pd.DataFrame: the parquet file loaded into a pandas.DataFrame.

    """

    # Get parquet file from GitHub
    headers = {}
    if token:
        headers['Authorization'] = f"token {token}"
    url = f'https://raw.githubusercontent.com/{username}/{repository_name}/main/'
    url = url + f'{file_path}/{file_name}'
    r = requests.get(url, headers=headers, timeout=200)
    r.raise_for_status()

    # Save a local copy of the dataset
    os.makedirs(file_path, exist_ok=True)
    local_file = os.path.join(file_path, file_name)
    with open(local_file, 'wb') as f:
        f.write(r.content)

    data = pd.read_parquet(local_file)

    return data


def test_github_read_parquet() -> None:
    """Test module."""
    df = github_read_parquet()
    print(df)


def test_dataset_from_mongo():
    """Test of 'dataset_from_mongo' function."""
    db_name = "datasets_flujo_cobranza"
    col_name = "v1"
    dataset_from_mongo(database_name=db_name, collection_name=col_name)


if __name__ == "__main__":
    test_github_read_parquet()
