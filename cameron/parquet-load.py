import pandas as pd
import glob
import requests
import json
import os  # Make sure to import os at the top of your file

def load_parquet_files(folder_path):
    # Expand the user's home directory
    folder_path = os.path.expanduser(folder_path)
    # Create a pattern to match all parquet files in the folder
    pattern = f"{folder_path}/*.parquet"
    # Use glob to find all files matching the pattern
    parquet_files = glob.glob(pattern)
    # Load and concatenate all parquet files into a single DataFrame
    df = pd.concat([pd.read_parquet(file) for file in parquet_files[0:1]], ignore_index=True)
    return df

def post_text_to_endpoint(df, url):
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract text from the current row
        text = row['text']
        # Prepare the data for POST request
        data = json.dumps({"text": text})
        headers = {'Content-Type': 'application/json'}
        # Send POST request to the specified URL
        response = requests.post(url, data=data, headers=headers)
        # Check if the request was successful
        if response.status_code == 200:
            print(f"Successfully posted text from row {index}")
        else:
            print(f"Failed to post text from row {index}. Status code: {response.status_code}")

# Define the folder path and URL
folder_path = "~/code/tiny-strange-textbooks"
url = "https://cpfiffer--mongo-hack-handle-request-dev.modal.run/"

# Load parquet files into a DataFrame
df = load_parquet_files(folder_path)
print(df)

# Post text to the specified endpoint
post_text_to_endpoint(df, url)
