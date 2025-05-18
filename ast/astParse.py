import javalang
import json

def print_node(node, indent=0):
    """Prints the AST node in a readable format."""
    spacing = ' ' * indent
    if isinstance(node, javalang.tree.Node):
        print(f"{spacing}{type(node).__name__}")
        for field, value in node.__dict__.items():
            if field == 'position':  # Skip position details for clarity
                continue
            print(f"{spacing}  {field}:")
            if isinstance(value, list):
                for item in value:
                    print_node(item, indent + 4)
            elif isinstance(value, javalang.tree.Node):
                print_node(value, indent + 4)
            else:
                print(f"{spacing}    {value}")
    else:
        print(f"{spacing}{node}")

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

# Example Java method (can be replaced with any input)
java_code = """
@Test
public void testResumeContainerEvent()
    throws IllegalArgumentException, IllegalAccessException, IOException {
  spy.running.clear();
  spy.running.put(containerId, containerLaunch);
  when(event.getType())
      .thenReturn(ContainersLauncherEventType.RESUME_CONTAINER);
  doNothing().when(containerLaunch).resumeContainer();
  spy.handle(event);
  assertEquals(1, spy.running.size());
  Mockito.verify(containerLaunch, Mockito.times(1)).resumeContainer();
}
"""

# Dynamically wrap if necessary
wrapped_code = wrap_in_class(java_code)

try:
    # Tokenize and parse the wrapped Java code
    tokens = javalang.tokenizer.tokenize(wrapped_code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse()

    print("Parsed successfully!")

    # Convert the parsed tree to a dictionary
    parsed_tree_dict = node_to_dict(tree)

    # Remove the temporary class from the parsed tree
    cleaned_tree = remove_temp_class(parsed_tree_dict)

    # Save the cleaned dictionary as a JSON file
    with open("parsed_cleaned_java_tree.json", "w", encoding="utf-8") as json_file:
        json.dump(cleaned_tree, json_file, indent=4, ensure_ascii=False)

    print("Parsed and cleaned Java syntax tree saved to 'parsed_cleaned_java_tree.json'.")
    print("---------------------------------------------------------")
    print_node(tree)

except javalang.parser.JavaSyntaxError as e:
    print(f"Syntax error while parsing: {e}")
except Exception as ex:
    print(f"An unexpected error occurred: {ex}")
