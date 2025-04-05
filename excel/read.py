import pandas as pd


def readRepoTestCases(file_path):
    # Path to your Excel file
    # file_path = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\eagerTest\spring-cloud-zuul-ratelimit.csv'

    # Read first 15 rows
    df = pd.read_csv(file_path)
    test_cases = df['testCase']

    for case in test_cases:
        print(case)

    return test_cases
