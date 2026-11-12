"""Check that both configs disable VarParsing's automatic output-name tags."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['nano_mc_cfg.py', 'nano_data_cfg.py'])
def test_plain_output_name_is_selected_before_parsing(name):
    source = (ROOT/'cmssw'/name).read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    setting = [node for node in calls if ast.unparse(node.func) == 'options.setType'
               and [ast.unparse(arg) for arg in node.args] == ["'outputFile'", 'VarParsing.varType.string']]
    parsing = [node for node in calls if ast.unparse(node.func) == 'options.parseArguments']
    assert len(setting) == len(parsing) == 1
    assert setting[0].lineno < parsing[0].lineno
