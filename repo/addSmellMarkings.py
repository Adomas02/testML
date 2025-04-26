import pandas as pd
import os


def addSmellMarking(testCodeFile, project_name):
    eagerTestPath = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\eagerTest'
    mysteryGuestPath = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\mysteryGuest'
    resourceOptimismPath = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\resourceOptimism'
    testRedundancyPath = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\testRedundancy'

    paths = [
        (os.path.join(eagerTestPath, project_name), 'isEagerTestManual'),
        (os.path.join(mysteryGuestPath, project_name), 'isMysteryGuestManual'),
        (os.path.join(resourceOptimismPath, project_name), 'isResourceOptimismManual'),
        (os.path.join(testRedundancyPath, project_name), 'isTestRedundancyManual')]

    merged_df = None

    for file_path, column_name in paths:
        print(file_path)
        print(column_name)


    # Loop through files
    for file_path, column_name in paths:
        df = pd.read_csv(file_path, usecols=['testCase', column_name])

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='testCase')

    merged_df.to_csv('combined_test_smells.csv', index=False)

    df1 = pd.read_csv('combined_test_smells.csv')

    df2 = pd.read_csv(testCodeFile)

    merged_df = pd.merge(df1, df2, on='testCase')

    merged_df.to_csv('merged_file.csv', index=False)
