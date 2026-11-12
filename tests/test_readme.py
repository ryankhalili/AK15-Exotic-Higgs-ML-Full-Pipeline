import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / 'README.md'


def bash_blocks():
    return re.findall(r'```bash\n(.*?)\n```', README.read_text(), flags=re.S)


def test_documented_script_paths_and_options():
    checked = set()
    for block in bash_blocks():
        joined = block.replace('\\\n', '')
        for script, arguments in re.findall(
                r'(?m)^(?:\s*ak15_combine )?\s*python(?:3)? '
                r'((?:scripts|tools)/\S+\.py)([^\n]*)', joined):
            path = ROOT / script
            assert path.is_file(), script
            tree = ast.parse(path.read_text())
            options = {
                arg.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                and arg.value.startswith('-')
            }
            for option in re.findall(r'(?<!\S)(--[a-z][a-z-]*)', arguments):
                assert option in options, (script, option)
            checked.add(script)
    assert len(checked) >= 12


def test_workflow_steps_and_local_links():
    text = README.read_text()
    steps = [int(n) for n in re.findall(r'^### (\d+)\.', text, flags=re.M)]
    assert steps == list(range(1, 17))
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if '://' in target or target.startswith('#'):
            continue
        assert (ROOT / target.split('#')[0]).is_file(), target


def test_embedded_python_examples_parse():
    examples = re.findall(r"python - <<'PY'\n(.*?)\nPY", README.read_text(), flags=re.S)
    assert len(examples) >= 2
    for code in examples:
        ast.parse(code)
