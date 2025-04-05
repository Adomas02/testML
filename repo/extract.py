import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from excel.read import readRepoTestCases


root_dir = r'C:\tests\spring-cloud-zuul-ratelimit'

fully_qualified_tests = readRepoTestCases(r'C:\Users\kazen\Desktop\ML-Test-Smell-Detection-Online-Appendix\dataset\eagerTest\spring-cloud-zuul-ratelimit.csv')

# Store results
found_tests = []

def extract_method_code(content, method_name):
    # Regex to find method start (case-insensitive)
    method_pattern = re.compile(
        r'(?:@Test\s+)?(?:public\s+)?[\w<>]+\s+' + re.escape(method_name) + r'\s*\([^)]*\)\s*\{',
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
            return content[start_index:i+1]
    return None


for test_path in fully_qualified_tests:
    parts = test_path.split('.')
    method_name = parts[-1]
    class_path = os.path.join(*parts[:-1]) + '.java'
    target_filename = os.path.basename(class_path)

    # Search for the Java file
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
                            found_tests.append((test_path, full_path, method_code.strip()))
                            break
                except (UnicodeDecodeError, FileNotFoundError):
                    continue

# Output
if found_tests:
    for test, path, code in found_tests:
        print(f"\n=== {test} in {path} ===\n")
        print(code)
else:
    print("No matching test methods found.")
