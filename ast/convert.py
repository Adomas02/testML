import pandas as pd
import javalang
import json

def node_to_dict(node):
    """Converts a javalang AST node to a dictionary recursively."""
    if isinstance(node, javalang.tree.Node):
        node_dict = {
            "type": type(node).__name__,
            "fields": {}
        }
        for field, value in node.__dict__.items():
            if field == 'position':  # Skip position details for clarity
                continue
            if isinstance(value, list):
                node_dict["fields"][field] = [node_to_dict(item) for item in value]
            elif isinstance(value, javalang.tree.Node):
                node_dict["fields"][field] = node_to_dict(value)
            else:
                node_dict["fields"][field] = str(value)
        return node_dict
    else:
        return str(node)

def wrap_in_class(java_code):
    """Dynamically wraps a Java method or fragment into a minimal class structure."""
    if "class" not in java_code:
        # Wrap the code in a minimal class structure
        wrapped_code = f"""
public class TempClass {{
{java_code}
}}
"""
        return wrapped_code
    return java_code

def remove_temp_class(parsed_tree):
    """Removes the temporary class node from the parsed AST dictionary."""
    # Navigate to the class declaration and return its members directly
    if parsed_tree.get("type") == "CompilationUnit":
        types = parsed_tree.get("fields", {}).get("types", [])
        if types and types[0].get("type") == "ClassDeclaration":
            # Return only the method declarations or class body inside the temporary class
            return types[0].get("fields", {}).get("body", [])
    return parsed_tree

def java_to_ast(java_code):
    """Converts Java code to its AST representation."""
    try:
        wrapped_code = wrap_in_class(java_code)
        tokens = javalang.tokenizer.tokenize(wrapped_code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse()
        ast_dict = node_to_dict(tree)
        cleaned_tree = remove_temp_class(ast_dict)
        return json.dumps(cleaned_tree, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

# Load the merged CSV file
file_path = r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file.csv'  # Adjust the file path as needed
data = pd.read_csv(file_path)

# Apply the AST conversion to the 'method_code' column
data['method_code'] = data['method_code'].astype(str).apply(java_to_ast)

# Save the updated DataFrame to a new CSV file
output_path = 'ast_merged_file.csv'
data.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"AST-based CSV file saved as: {output_path}")
