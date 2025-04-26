import pandas as pd


def readRepoTestCases(file_path):

    df = pd.read_csv(file_path)
    test_cases = df['testCase']

    # for case in test_cases:
    #     print(case)

    return test_cases
