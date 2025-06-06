import pandas as pd
from colorama import Fore, Style
from google.cloud import bigquery
from params import *

def get_data(data_path) -> pd.DataFrame:

    df = pd.read_csv(data_path)
    df["index"]=df.index
    return df


def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """
    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)
    client = bigquery.Client(project=gcp_project)
    if truncate == True:
        write_mode = "WRITE_TRUNCATE"
    else:
        write_mode = "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

# path = "raw_data/HAM10000_metadata.csv"
# #path = "raw_data/hmnist_28_28_RGB.csv"

# load_data_to_bq(get_data(path),f"{GCP_PROJECT}",f"{BQ_DATASET}","train_metadata",True)


import os
from google.cloud import storage

def upload_images_folder(local_folder, bucket_name, destination_folder):
    if not os.path.exists(local_folder):
        print(f"❌ Folder not found: {local_folder}")
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    uploaded_count = 0

    for filename in os.listdir(local_folder):
        if filename.lower().endswith('.jpg'):
            local_path = os.path.join(local_folder, filename)
            blob_path = os.path.join(destination_folder, filename)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            uploaded_count += 1
            print(f"✅ Uploaded: {filename}, {uploaded_count}")

    print(f"\n🎉 Uploaded {uploaded_count} .jpg files to '{destination_folder}' in bucket '{bucket_name}'")

#upload_images_folder("raw_data/test_data/ISIC2018_Task3_Test_Input", BUCKET_NAME, "test_all_images")
def get_metadata_from_bq(
        gcp_project="skin-scan-461716",
        query="""
        SELECT *
        FROM `skin-scan-461716.skin_scan.train_metadata`
        ORDER BY image_id
    """,
    ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery
    """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df

query_metadata = f"""
        SELECT *
        FROM `skin-scan-461716.skin_scan.train_metadata`
    """
