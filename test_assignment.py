import numpy as np
import pandas as pd

from base_classes import CSVDataHandler
from ideal_function_selector import FunctionSelector
from mapping import Test_Mapper


def test_csvhandler_load_and_validate(tmp_path):
    p = tmp_path / "sample.csv"
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    df.to_csv(p, index=False)

    handler = CSVDataHandler(str(p))
    loaded = handler.load_data()
    assert len(loaded) == 3
    assert "x" in loaded.columns and "y" in loaded.columns

    # validation with required columns
    assert handler.validate_data(["x", "y"]) is True
    assert handler.data.equals(loaded)


def test_functionmatcher_basic():
    x = np.linspace(0, 1, 5)
    training = pd.DataFrame({
        "x": x,
        "y1": 2 * x,
        "y2": 3 * x,
        "y3": -x,
        "y4": 0 * x,
    })

    ideal = pd.DataFrame({"x": x})
    ideal["y1"] = 2 * x
    ideal["y2"] = 3 * x
    ideal["y3"] = -x
    ideal["y4"] = 0 * x

    # add extra ideal cols
    for i in range(5, 11):
        ideal[f"y{i}"] = np.random.randn(len(x))

    matcher = FunctionSelector(training, ideal)
    selected = matcher.select_all_functions()

    assert isinstance(selected, dict)
    assert len(selected) == 4
    assert all(isinstance(v, int) for v in selected.values())


def test_simplemapper_map_all():
    x = np.linspace(0, 10, 11)
    ideal = pd.DataFrame({"x": x})
    ideal["y1"] = x
    ideal["y2"] = 2 * x
    for i in range(3, 6):
        ideal[f"y{i}"] = np.random.randn(len(x))

    test_df = pd.DataFrame({"x": [2.0, 5.0, 8.0], "y": [2.1, 10.1, 15.9]})

    selected = {1: 1, 2: 2, 3: 3, 4: 4}
    max_devs = {1: 0.5, 2: 0.5, 3: 1.0, 4: 1.0}

    mapper = Test_Mapper(test_df, ideal, selected, max_devs)
    mappings = mapper.map_all_test_points()

    # mappings should have expected columns (may be empty)
    assert all(col in mappings.columns for col in ["x", "y", "delta_y", "ideal_func_no"]) or mappings.empty
    if not mappings.empty:
        assert (mappings["delta_y"] >= 0).all()
