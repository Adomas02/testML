import pandas as pd

# Load the CSV
df = pd.read_csv(r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file.csv')

# Function to count True values (works for boolean and string 'True')
def count_true(col):
    return (col == True).sum() + (col == 'True').sum()

# Count True values per column
true_counts = df.apply(count_true)

# Count non-null values per column for percentage calculation
non_null_counts = df.notnull().sum()

# Calculate the percentage of True values per column
percentages = (true_counts / non_null_counts) * 100

# Format results as a pretty DataFrame
results = pd.DataFrame({
    'Column': true_counts.index,
    'True_Count': true_counts.values,
    'Percentage_True': percentages.values.round(2)
})

# Sort by count or percentage if desired
results = results.sort_values(by='True_Count', ascending=False)

# Show results
print(results)

# Save to CSV
results.to_csv(r'C:\Users\kazen\PycharmProjects\testML\repo\true_counts_percentages_per_column.csv', index=False)

