import pandas as pd


def addSmellMarking(testCodeFile):
    paths = [(r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\eagerTest\spring-cloud-zuul-ratelimit.csv', 'isEagerTestManual'),
            ( r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\mysteryGuest\spring-cloud-zuul-ratelimit.csv','isMysteryGuestManual'),
             (r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\resourceOptimism\spring-cloud-zuul-ratelimit.csv','isResourceOptimismManual'),
             (r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\testRedundancy\spring-cloud-zuul-ratelimit.csv','isTestRedundancyManual')]

    merged_df = None

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






