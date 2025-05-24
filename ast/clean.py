import pandas as pd

# Load the file with ASTs
df = pd.read_csv(r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file_with_ast.csv')

# Remove rows with empty method_code or method_ast
df = df.dropna(how='all')
df = df[~df.apply(lambda row: all((str(x).strip() == "" or pd.isna(x)) for x in row), axis=1)]

# Remove rows where method_ast contains "error"
df = df[~df["method_ast"].str.contains('"error"', na=False)]

# Reset index
df = df.reset_index(drop=True)

# Save cleaned file
cleaned_path = r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file_with_ast_cleaned.csv'
df.to_csv(cleaned_path, index=False)