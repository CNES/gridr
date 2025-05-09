import os
import re

# Path to the Python source code
SRC_DIR = "../../python/gridr"
# Path to generate the documentation
DOCS_DIR = "api_python"

# Tag used to filter documented members
DOC_TAG = "@doc"
ENABLE_TAG_FILTERING=False

def check_subpackage(dir_path):
    if dir_path.endswith("__"):
        return False
    for f in os.listdir(dir_path):
        if f.endswith(".py"):
            return False
    if len(os.listdir(dir_path)) == 0:
        return False
    return True

def find_python_modules(base_path):
    """
    Traverse the directory and return a list of Python modules while preserving hierarchy.
    """
    modules = []
    subpackages = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_path = os.path.join(root, file)
                module_name = os.path.relpath(module_path, base_path).replace("/", ".").replace(".py", "")
                modules.append(module_name)
        # Check for directories to create index.rst for sub-packages
        for cdir in dirs:
            dir_path = os.path.join(root, cdir)
            if check_subpackage(dir_path):  # No Python files in the directory
                
                submodule_name = os.path.relpath(dir_path, base_path).replace("/", ".")
                subpackages.append(submodule_name)
                
    return modules, subpackages

def extract_documented_members(filepath):
    """
    Extracts classes and functions/methods that contain @doc in their docstring.
    """
    documented_members = []
    if ENABLE_TAG_FILTERING:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Search for classes and functions/methods with @doc in their docstring
        pattern = re.compile(r"^(class|def) (\w+).*?:\s*\"\"\"(.*?)\"\"\"", re.DOTALL | re.MULTILINE)
        for match in pattern.finditer(content):
            if DOC_TAG in match.group(3):
                documented_members.append(match.group(2))

    return documented_members

def generate_rst_files(base_path, doc_path):
    """
    Generates .rst files while preserving the code structure and filtering members.
    """
    os.makedirs(doc_path, exist_ok=True)
    modules, subpackages = find_python_modules(base_path)

    index_entries = set()

    for module in modules:
        module_path = os.path.join(doc_path, module.replace(".", "/") + ".rst")
        os.makedirs(os.path.dirname(module_path), exist_ok=True)

        # Path to the corresponding Python source file
        py_file = os.path.join(base_path, module.replace(".", "/") + ".py")

        # Extract filtered members
        documented_members = extract_documented_members(py_file)

        with open(module_path, "w", encoding="utf-8") as f:
            f.write(f"{module}\n{'=' * len(module)}\n\n")
            f.write(f".. automodule:: {os.path.basename(SRC_DIR)}.{module}\n")
            #f.write("    :members:\n    :undoc-members:\n    :show-inheritance:\n\n")
            if documented_members:
                f.write("    :members: " + ", ".join(documented_members) + "\n")
            else:
                f.write("    :members: \n")

        # Add to the corresponding index file
        index_dir = os.path.dirname(module_path)
        index_entries.add(index_dir)
    
    for subpackage in subpackages:
        
        subpackage_path = os.path.join(SRC_DIR, subpackage.replace(".", "/"))
        subpackage_index = os.path.join(doc_path, subpackage.replace(".", "/") + "/index.rst")
        children_subpackage = [
            os.path.splitext(f)[0]
            for f in os.listdir(subpackage_path)
            if len(os.listdir(os.path.join(subpackage_path, f))) > 0
        ]
        with open(subpackage_index, "w", encoding="utf-8") as f:
            f.write(f"{subpackage}\n{'=' * len(subpackage)}\n\n")
            f.write(".. toctree::\n    :maxdepth: 2\n\n")
            for sub in sorted(children_subpackage):
                f.write(f"    {sub}/index\n")

    return index_entries

def create_index_files(index_entries):
    """
    Generates index.rst files for each relevant directory.
    """
    for index_dir in index_entries:
        module_name = os.path.relpath(index_dir, DOCS_DIR).replace("/", ".")
        submodules = [
            os.path.splitext(f)[0]
            for f in os.listdir(index_dir)
            if f.endswith(".rst") and f != "index.rst"
        ]

        index_file = os.path.join(index_dir, "index.rst")
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(f"{module_name}\n{'=' * len(module_name)}\n\n")
            f.write(".. toctree::\n    :maxdepth: 2\n\n")
            for sub in sorted(submodules):
                f.write(f"    {sub}\n")

def generate_modules_rst():
    """
    Creates the modules.rst file inside api_python/.
    """
    modules_rst = os.path.join(DOCS_DIR, "modules.rst")  # Now inside api_python/
    with open(modules_rst, "w", encoding="utf-8") as f:
        f.write("API Documentation\n=================\n\n")
        f.write(".. toctree::\n    :maxdepth: 2\n    :caption: API\n\n")
        f.write("    chain/index\n")
        f.write("    core/index\n")
        f.write("    io/index\n")
        f.write("    scaling/index\n")

# Execute the script
index_entries = generate_rst_files(SRC_DIR, DOCS_DIR)
create_index_files(index_entries)
generate_modules_rst()

print("✅ Sphinx documentation generated successfully!")
