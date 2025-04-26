import os
import re
import sys
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from excel.read import readRepoTestCases
from repo.addSmellMarkings import addSmellMarking


def extract_method_code(content, method_name):
    # Regex to find method start (case-insensitive)
    method_pattern = re.compile(
        r'(?:@Test\s+)?(?:public\s+)?[\w<>]+\s+' + re.escape(method_name) +
        r'\s*\([^)]*\)\s*(?:throws\s+\w+(?:\s*,\s*\w+)*)?\s*\{',
        re.IGNORECASE
    )
    match = method_pattern.search(content)
    if not match:
        return None

    start_index = match.start()
    brace_count = 0
    inside = False
    for i in range(start_index, len(content)):
        if content[i] == '{':
            brace_count += 1
            inside = True
        elif content[i] == '}':
            brace_count -= 1
        if inside and brace_count == 0:
            return content[start_index:i + 1]
    return None



# Your target folder
folder_path = r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\eagerTest'  # <-- Change this

# List all CSV files with full paths
listRepos = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))
]

# Print all CSV files
# for file in listRepos:
#     print(file)

# listRepos = [r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\eagerTest\spring-cloud-zuul-ratelimit.csv',
#              r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\eagerTest\achilles.csv']

for repo in listRepos:
    filename = os.path.splitext(os.path.basename(repo))[0]
    root_dir = os.path.join(r'C:\tests', filename)
    print(root_dir)
    fully_qualified_tests = readRepoTestCases(repo)

    # Store results
    found_tests = []

    for test_path in fully_qualified_tests:
        parts = test_path.split('.')
        method_name = parts[-1]
        class_path = os.path.join(*parts[:-1]) + '.java'
        target_filename = os.path.basename(class_path)

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:

                if filename.lower() == target_filename.lower():
                    full_path = os.path.join(dirpath, filename)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            # Use regex to search for the method (loose pattern to avoid false negatives)
                            method_code = extract_method_code(content, method_name)

                            if method_code:
                                found_tests.append((test_path, method_code.strip()))
                                break
                    except (UnicodeDecodeError, FileNotFoundError):
                        continue

    if found_tests:
        # for test_path, code in found_tests:
        #     print(test_path)
        #     print(code)
        #     print("----------------------------------")

        with open('found_tests.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            writer.writerow(["testCase", "method_code"])

            for row in found_tests:
                writer.writerow(row)
    else:
        print("No matching test methods found.")

    addSmellMarking('found_tests.csv',repo)





