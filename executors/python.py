import subprocess
import sys
import tempfile
import os
import json

def execute_python_code(code: str) -> str:
    """Execute Python code and return the output."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Clean up the temporary file
        os.unlink(temp_file)
        
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        else:
            return result.stdout
    
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out"
    except Exception as e:
        return f"Error: {str(e)}"

def execute_function(function_string: str, inputs: dict):
    """Execute a function with given inputs and return the output."""
    try:
        # Create test code
        test_code = f"""{function_string}

import json
import ast

# Get function name
tree = ast.parse('''{function_string}''')
function_name = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        function_name = node.name
        break

if function_name:
    inputs = {json.dumps(inputs)}
    try:
        result = globals()[function_name](**inputs)
        print(json.dumps(result))
    except Exception as e:
        print(f"Error: {{e}}")
else:
    print("Error: No function found")
"""
        
        return execute_python_code(test_code)
    
    except Exception as e:
        return f"Error: {str(e)}"