import pandas as pd
import os
from git import Repo

# 1. Load your CSV file
csv_path = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\projects.csv'

# Read the CSV
df = pd.read_csv(csv_path)

# 2. Assume repo links are in a column called 'repo_link'
repo_links = df['url'].dropna().tolist()

# 3. Folder to clone repos into
base_clone_folder = r'C:\tests'
os.makedirs(base_clone_folder, exist_ok=True)  # Create folder if it doesn't exist

# 4. Loop and clone each repo
for url in repo_links:
    repo_name = url.rstrip('/').split('/')[-1]  # Get last part of URL for folder
    clone_path = os.path.join(base_clone_folder, repo_name)

    if os.path.exists(clone_path):
        print(f"Repository '{repo_name}' already exists, skipping...")
    else:
        print(f"Cloning '{repo_name}'...")
        try:
            Repo.clone_from(url, clone_path)
            print(f"'{repo_name}' cloned successfully!")
        except Exception as e:
            print(f"Failed to clone '{repo_name}': {e}")

print("✅ All repositories processed.")
