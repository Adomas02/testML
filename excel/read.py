import pandas as pd


def readRepoTestCases(file_path):

    df = pd.read_csv(file_path)
    test_cases = df['testCase']



    return test_cases
