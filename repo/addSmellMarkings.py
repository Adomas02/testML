import pandas as pd
import os


def addSmellMarking(testCodeFile, project_name):
    base_project_name = os.path.basename(project_name)
    eagerTestPath = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\eagerTest'
    mysteryGuestPath = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\mysteryGuest'
    resourceOptimismPath = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\resourceOptimism'
    testRedundancyPath = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\testRedundancy'

    paths = [
        (os.path.join(eagerTestPath, base_project_name), 'isEagerTestManual'),
        (os.path.join(mysteryGuestPath, base_project_name), 'isMysteryGuestManual'),
        (os.path.join(resourceOptimismPath, base_project_name), 'isResourceOptimismManual'),
        (os.path.join(testRedundancyPath, base_project_name), 'isTestRedundancyManual')]

    merged_df = None

    # Loop through files
    for file_path, column_name in paths:
        df = pd.read_csv(file_path, usecols=['testCase', column_name])

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='testCase')

    df_test_code = pd.read_csv(testCodeFile)

    project_merged_df = pd.merge(merged_df, df_test_code, on='testCase')

    # Check if merged_file.csv exists already
    if os.path.exists('merged_file.csv'):
        # Load existing merged_file.csv
        existing_df = pd.read_csv('merged_file.csv')
        # Append the new data
        combined_df = pd.concat([existing_df, project_merged_df], ignore_index=True)
        # Drop duplicates if same testCase shows up again (optional)
        combined_df = combined_df.drop_duplicates(subset=['testCase'])
        # Save back
        combined_df.to_csv('merged_file.csv', index=False)
    else:
        # No file exists yet, create it
        project_merged_df.to_csv('merged_file.csv', index=False)
