import javalang
import ast
import json


def print_node(node, indent=0):
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

# Example Java code (you can replace this with the contents of your test file)
java_code = """
import org.junit.Test;
import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

public class TestClass {
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
}
"""

tokens = javalang.tokenizer.tokenize(java_code)
parser = javalang.parser.Parser(tokens)
# Parsing the entire compilation unit
try:
    tree = parser.parse()
    print("Parsed successfully!")
    # print(tree)

    # Convert the parsed tree to a dictionary
    parsed_tree_dict = node_to_dict(tree)

    # Save the dictionary as a JSON file
    with open("parsed_java_tree.json", "w", encoding="utf-8") as json_file:
        json.dump(parsed_tree_dict, json_file, indent=4)
    print("Parsed Java syntax tree saved to 'parsed_java_tree.json'.")
    print("---------------------------------------------------------")
    # print_node(tree)

except javalang.parser.JavaSyntaxError as e:
    print(f"Syntax error: {e}")



# Parse the Java code
# tree = javalang.parse.parse(java_code)
# print(res)

#
# # Print the AST structure
# for path, node in tree:
#     print(f"{'  ' * len(path)}{type(node).__name__}")
