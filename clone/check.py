import os
import pandas as pd
# Path where you want to list directories
parent_folder = r'C:\tests'  # <-- CHANGE THIS

# List all directories inside
directories = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]

# Print the list
# for folder in directories:
#     print(folder)


print("---------------")

csv_path = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\projects.csv'

# Read the CSV
df = pd.read_csv(csv_path)
repo_links = df['url'].dropna().tolist()

for link in repo_links:
    repo_name = link.rstrip('/').split('/')[-1]
    # print(repo_name)

    if repo_name not in directories:
        print(repo_name)

