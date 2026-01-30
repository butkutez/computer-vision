import os
import kagglehub
from dotenv import load_dotenv
import shutil

# 1. Environment Setup: Load API credentials from .env file
load_dotenv()
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

# 2. Data Acquisition: Download dataset from Kaggle
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)

# 3. Target directory for project data
local_data_dir = "data/"

# 4. Clean up and Copy
if os.path.exists(local_data_dir):
    print("Empty folder detected. Removing and restarting copy...")
    shutil.rmtree(local_data_dir)

print("Copying files... this will take a moment (2.3GB)")
shutil.copytree(path, local_data_dir)
print("Done! Verify now with 'ls data/'")

