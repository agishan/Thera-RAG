import sys
from pathlib import Path
import types
import pytest

# Provide a minimal pandas stub so content_utils can be imported without pandas
class _SimpleDataFrame:
    def __init__(self, data, columns=None):
        self._data = [list(row) for row in data]
        self.columns = list(columns) if columns is not None else []
    @property
    def empty(self):
        return len(self._data) == 0
    class _ILoc:
        def __init__(self, outer):
            self._outer = outer
        def __getitem__(self, idx):
            class _Row(list):
                def tolist(self):
                    return list(self)
            return _Row(self._outer._data[idx])
    @property
    def iloc(self):
        return _SimpleDataFrame._ILoc(self)

pd_stub = types.ModuleType('pandas')
pd_stub.DataFrame = _SimpleDataFrame
sys.modules.setdefault('pandas', pd_stub)

# Minimal streamlit stub for importing content_utils
st_stub = types.ModuleType('streamlit')

def _no_op(*args, **kwargs):
    pass

st_stub.markdown = _no_op
st_stub.code = _no_op
st_stub.dataframe = _no_op
sys.modules.setdefault('streamlit', st_stub)

# Add src/app to path for importing content_utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from content_utils import parse_markdown_table


def test_parse_valid_markdown_table():
    table_text = """
    | Name | Age |
    |------|-----|
    | Alice | 30 |
    | Bob   | 25 |
    """
    df = parse_markdown_table(table_text)
    assert df is not None
    assert df.columns == ["Name", "Age"]
    assert df.iloc[0].tolist() == ["Alice", "30"]
    assert df.iloc[1].tolist() == ["Bob", "25"]


def test_parse_invalid_or_empty_table_returns_none():
    assert parse_markdown_table("") is None
    assert parse_markdown_table("|----|----|") is None
