import ast
import os
path = os.getcwd() + "/"

# getting_started.rst creation
with open("getting_started.rst", "w") as rst_file:
    s = "Getting Started"
    print(s, file=rst_file)
    print(len(s) * "=", file=rst_file)
    print(".. toctree::\n", file=rst_file)
    print("   getting_started/installation\n   getting_started/tutorials\n   getting_started/contributors"
          "\n   getting_started/publications\n ", file=rst_file)


# extract function names from .py files
def extract_function_names(file_path):
    function_names = []
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
    return function_names


# rst4conversions
def rst4conv(module, rst_name, names):
    p = path + module
    try:
        os.mkdir(p)
    except FileExistsError:
        pass
    os.chdir(p)
    with open(rst_name + ".rst", "w") as rst_file:
        print(rst_name, file=rst_file)
        print(len(rst_name) * "=", file=rst_file)
        print(".. currentmodule:: gunfolds." + module, file=rst_file)
        print("\n", file=rst_file)
        function_names = names
        function_names.sort(key = lambda x:x.lower())
        for func in function_names:
            funct = ""
            for c in func:
                if c == "_":
                    funct += "\\"
                    funct += c
                else:
                    funct += c
            print(funct, file=rst_file)
            print(len(funct) * "-", file=rst_file)
            print(".. autofunction:: gunfolds." + module + "." + func, file=rst_file)
            print("\n", file=rst_file)
    os.chdir("..")


for filename in os.listdir("../gunfolds"):
    if filename == "conversions.py":
        module = filename[:-3]
        with open("conversions.rst", "w") as rst_file:
            s = "conversions"
            print(s, file=rst_file)
            print(len(s) * "=", file=rst_file)
            print(".. toctree::\n", file=rst_file)
            print("   conversions/clingo", file=rst_file)
            print("   conversions/external", file=rst_file)
            print("   conversions/internal", file=rst_file)
        function_names = extract_function_names("../gunfolds/conversions.py")
        indx = []
        for i in range(len(function_names)):
            if function_names[i]=="sphinx_automation_fake":
                indx.append(i)
        graph_conv = function_names[0:indx[0]]
        clingo_conv = function_names[indx[0]+1:indx[1]]
        other_conv = function_names[indx[1]+1:]
        rst4conv(module, "internal", graph_conv)
        rst4conv(module, "clingo", clingo_conv)
        rst4conv(module, "external", other_conv)


# rst4estimation
def rst4estimation(module):
    p = path + "estimation"
    try:
        os.mkdir(p)
    except FileExistsError:
        pass
    os.chdir(p)
    with open(module + ".rst", "w") as rst_file:
        print(module, file=rst_file)
        print(len(module) * "=", file=rst_file)
        print(".. currentmodule:: gunfolds.estimation." + module, file=rst_file)
        print("\n", file=rst_file)
        file_path = "../../gunfolds/estimation"
        function_names = extract_function_names(file_path + "/" + module + ".py")
        function_names.sort(key = lambda x:x.lower())
        for func in function_names:
            funct = ""
            for c in func:
                if c == "_":
                    funct += "\\"
                    funct += c
                else:
                    funct += c
            print(funct, file=rst_file)
            print(len(funct) * "-", file=rst_file)
            print(".. autofunction:: gunfolds.estimation." + module + "." + func, file=rst_file)
            print("\n", file=rst_file)
    os.chdir("..")


for filename in os.listdir("../gunfolds/estimation"):
    if filename.endswith(".py") and not filename.endswith("__.py"):
        module = filename[:-3]
        rst4estimation(module)

with open("estimation" + ".rst", "w") as rst_file:
    s = "estimation"
    print(s, file=rst_file)
    print(len(s) * "=", file=rst_file)
    print(".. toctree::\n", file=rst_file)
    li = os.listdir("../gunfolds/estimation")
    li.sort(key = lambda x:x.lower())
    for filename in li:
        if filename.endswith(".py") and not filename.endswith("__.py"):
            print("   estimation/" + filename[:-3], file=rst_file)


# rst4solvers
def rst4solvers(module):
    p = path + "solvers"
    try:
        os.mkdir(p)
    except FileExistsError:
        pass
    os.chdir(p)
    if module == "clingo_msl" or module == "clingo_rasl":
        try:
            os.mkdir(p + "/declarative")
        except FileExistsError:
            pass
        os.chdir(p + "/declarative")
    else:
        try:
            os.mkdir(p + "/imperative")
        except FileExistsError:
            pass
        os.chdir(p + "/imperative")
    with open(module + ".rst", "w") as rst_file:
        print(module, file=rst_file)
        print(len(module) * "=", file=rst_file)
        print(".. currentmodule:: gunfolds.solvers." + module, file=rst_file)
        print("\n", file=rst_file)
        file_path = "../../../gunfolds/solvers"
        function_names = extract_function_names(file_path + "/" + module + ".py")
        function_names.sort(key = lambda x:x.lower())
        for func in function_names:
            funct = ""
            for c in func:
                if c == "_":
                    funct += "\\"
                    funct += c
                else:
                    funct += c
            print(funct, file=rst_file)
            print(len(funct) * "-", file=rst_file)
            print(".. autofunction:: gunfolds.solvers." + module + "." + func, file=rst_file)
            print("\n", file=rst_file)
    os.chdir("../..")


for filename in os.listdir("../gunfolds/solvers"):
    if filename.endswith(".py") and not filename.endswith("__.py"):
        module = filename[:-3]
        rst4solvers(module)

with open("solvers" + ".rst", "w") as rst_file:
    s = "solvers"
    print(s, file=rst_file)
    print(len(s) * "=", file=rst_file)
    print(".. toctree::\n", file=rst_file)
    print("   solvers/declarative", file=rst_file)
    print("   solvers/imperative", file=rst_file)

os.chdir("solvers")

with open("declarative" + ".rst", "w") as rst_file:
    s = "declarative"
    print(s, file=rst_file)
    print(len(s) * "=", file=rst_file)
    print(".. toctree::\n", file=rst_file)
    print("   declarative/clingo_msl", file=rst_file)
    print("   declarative/clingo_rasl", file=rst_file)

with open("imperative" + ".rst", "w") as rst_file:
    s = "imperative"
    print(s, file=rst_file)
    print(len(s) * "=", file=rst_file)
    print(".. toctree::\n", file=rst_file)
    print("   imperative/traversal", file=rst_file)
    print("   imperative/unknownrate", file=rst_file)

os.chdir("..")

# rst4utils
def rst4utils(module):
    p = path + "utils"
    try:
        os.mkdir(p)
    except FileExistsError:
        pass
    os.chdir(p)
    with open(module + ".rst", "w") as rst_file:
        print(module, file=rst_file)
        print(len(module) * "=", file=rst_file)
        print(".. currentmodule:: gunfolds.utils." + module, file=rst_file)
        print("\n", file=rst_file)
        file_path = "../../gunfolds/utils"
        function_names = extract_function_names(file_path + "/" + module + ".py")
        function_names.sort(key = lambda x:x.lower())
        for func in function_names:
            funct = ""
            for c in func:
                if c == "_":
                    funct += "\\"
                    funct += c
                else:
                    funct += c
            print(funct, file=rst_file)
            print(len(funct) * "-", file=rst_file)
            print(".. autofunction:: gunfolds.utils." + module + "." + func, file=rst_file)
            print("\n", file=rst_file)
    os.chdir("..")


for filename in os.listdir("../gunfolds/utils"):
    if filename.endswith(".py") and not filename.endswith("__.py"):
        module = filename[:-3]
        rst4utils(module)

with open("utils" + ".rst", "w") as rst_file:
    s = "utils"
    print(s, file=rst_file)
    print(len(s) * "=", file=rst_file)
    print(".. toctree::\n", file=rst_file)
    li = os.listdir("../gunfolds/utils")
    li.sort(key = lambda x:x.lower())
    for filename in li:
        if filename.endswith(".py") and not filename.endswith("__.py"):
            print("   utils/" + filename[:-3], file=rst_file)


# rst4viz
def rst4viz(module):
    p = path + "viz"
    try:
        os.mkdir(p)
    except FileExistsError:
        pass
    os.chdir(p)
    with open(module + ".rst", "w") as rst_file:
        print(module, file=rst_file)
        print(len(module) * "=", file=rst_file)
        print(".. currentmodule:: gunfolds.viz." + module, file=rst_file)
        print("\n", file=rst_file)
        file_path = "../../gunfolds/viz"
        function_names = extract_function_names(file_path + "/" + module + ".py")
        function_names.sort(key = lambda x:x.lower())
        for func in function_names:
            funct = ""
            for c in func:
                if c == "_":
                    funct += "\\"
                    funct += c
                else:
                    funct += c
            print(funct, file=rst_file)
            print(len(funct) * "-", file=rst_file)
            print(".. autofunction:: gunfolds.viz." + module + "." + func, file=rst_file)
            print("\n", file=rst_file)
    os.chdir("..")


for filename in os.listdir("../gunfolds/viz"):
    if filename.endswith(".py") and not filename.endswith("__.py"):
        module = filename[:-3]
        rst4viz(module)

with open("viz" + ".rst", "w") as rst_file:
    s = "viz"
    print(s, file=rst_file)
    print(len(s) * "=", file=rst_file)
    print(".. toctree::\n", file=rst_file)
    li = os.listdir("../gunfolds/viz")
    li.sort(key = lambda x:x.lower())
    for filename in li:
        if filename.endswith(".py") and not filename.endswith("__.py"):
            print("   viz/" + filename[:-3], file=rst_file)