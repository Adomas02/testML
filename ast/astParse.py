import pandas as pd
import javalang
import json

# Load the file
df = pd.read_csv(r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file.csv')

# Function to recursively convert javalang AST nodes to JSON-serializable dicts
def ast_to_dict(node):
    if isinstance(node, javalang.ast.Node):
        result = {"_type": type(node).__name__}
        for field in node.attrs:
            value = getattr(node, field)
            result[field] = ast_to_dict(value)
        return result
    elif isinstance(node, list):
        return [ast_to_dict(item) for item in node]
    elif isinstance(node, (str, int, float, bool)) or node is None:
        return node
    else:
        return str(node)

# Parse and convert to AST for each row
asts = []
for code in df['method_code']:
    try:
        # Wrap method in a dummy class for parsing if necessary
        if code.strip().startswith("@") or code.strip().startswith("public") or code.strip().startswith("void"):
            wrapper = f"public class DummyClass {{ {code} }}"
        else:
            wrapper = code

        tree = javalang.parse.parse(wrapper)
        # Extract the method(s) node(s) from the parsed class
        if tree.types and hasattr(tree.types[0], 'body'):
            methods = [m for m in tree.types[0].body if isinstance(m, javalang.tree.MethodDeclaration)]
            ast_json = [ast_to_dict(m) for m in methods]
        else:
            ast_json = []
    except Exception as e:
        ast_json = {"error": str(e)}
    asts.append(json.dumps(ast_json))

# Add new column to dataframe
df['method_ast'] = asts

# Save updated file
output_path = r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file_with_ast.csv'
df.to_csv(output_path, index=False)

# import ace_tools as tools; tools.display_dataframe_to_user(name="Java Methods with AST", dataframe=df)
#
# output_path
