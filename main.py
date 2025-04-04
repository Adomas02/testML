
print("Hello")

import os

import os

def print_java_files_content(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                print(f"\n--- {file_path} ---")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(content)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

# Use raw string to avoid issues with backslashes
print_java_files_content(r'C:\tests\testTyrus')

