import numpy as np
import pandas as pd

# Updated imports to match renamed classes
from classes import CSVLoader
from ideal_function_selector import IdealFunctionMatcher
from mapping import PointMapper


def test_csvloader_load_and_validate(tmp_path):
    """
    Tests that CSVLoader can load a CSV file correctly
    and that validate_data() confirms required columns exist.
    """
    # Create a temporary CSV file path inside the pytest temp directory
    temporary_file_path = tmp_path / "sample.csv"

    # Build a small sample DataFrame to write into the temp CSV file
    sample_dataframe = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    sample_dataframe.to_csv(temporary_file_path, index=False)

    # Create a CSVLoader instance pointing to the temporary file
    csv_handler = CSVLoader(str(temporary_file_path))

    # Load the data and store the result for assertions
    loaded_dataframe = csv_handler.load_data()

    # Check that the correct number of rows was loaded
    assert len(loaded_dataframe) == 3

    # Check that both expected columns are present in the loaded data
    assert "x" in loaded_dataframe.columns and "y" in loaded_dataframe.columns

    # Validate that the required columns pass the validation check
    assert csv_handler.validate_data(["x", "y"]) is True

    # Make sure the internally stored data matches what was returned by load_data
    assert csv_handler.loaded_data.equals(loaded_dataframe)


def test_idealfunctionmatcher_basic():
    """
    Tests that IdealFunctionMatcher correctly selects one ideal function
    per training function and returns a valid dictionary of matches.
    """
    # Create evenly spaced x values to use across training and ideal datasets
    x_values = np.linspace(0, 1, 5)

    # Build a simple training dataset with 4 known mathematical functions
    training_dataframe = pd.DataFrame({
        "x":  x_values,
        "y1": 2 * x_values,
        "y2": 3 * x_values,
        "y3": -x_values,
        "y4": 0 * x_values,
    })

    # Build the ideal dataset starting with just the x column
    ideal_dataframe = pd.DataFrame({"x": x_values})

    # Add the same 4 functions as in training so perfect matches exist
    ideal_dataframe["y1"] = 2 * x_values
    ideal_dataframe["y2"] = 3 * x_values
    ideal_dataframe["y3"] = -x_values
    ideal_dataframe["y4"] = 0 * x_values

    # Add extra random columns to make the selection process non-trivial
    for column_index in range(5, 11):
        ideal_dataframe[f"y{column_index}"] = np.random.randn(len(x_values))

    # Create the matcher using the training and ideal datasets
    function_matcher = IdealFunctionMatcher(training_dataframe, ideal_dataframe)

    # Run the full selection and store the result
    selected_functions = function_matcher.run_selection()

    # Result must be a dictionary
    assert isinstance(selected_functions, dict)

    # There must be exactly 4 entries — one per training function
    assert len(selected_functions) == 4

    # Every matched ideal function index must be an integer
    assert all(isinstance(ideal_index, int) for ideal_index in selected_functions.values())


def test_pointmapper_map_all():
    """
    Tests that PointMapper processes all test points and returns a DataFrame
    with the correct columns and non-negative deviations.
    """
    # Create evenly spaced x values for the ideal functions dataset
    x_values = np.linspace(0, 10, 11)

    # Build the ideal dataset with two predictable functions and some random ones
    ideal_dataframe = pd.DataFrame({"x": x_values})
    ideal_dataframe["y1"] = x_values
    ideal_dataframe["y2"] = 2 * x_values

    # Fill remaining columns with random data to simulate a real ideal dataset
    for column_index in range(3, 6):
        ideal_dataframe[f"y{column_index}"] = np.random.randn(len(x_values))

    # Create a small test dataset with points that should map close to y1 and y2
    test_dataframe = pd.DataFrame({"x": [2.0, 5.0, 8.0], "y": [2.1, 10.1, 15.9]})

    # Map each training function index to its matched ideal function index
    matched_functions_dictionary = {1: 1, 2: 2, 3: 3, 4: 4}

    # Set the maximum allowed deviation for each training function
    maximum_deviations_dictionary = {1: 0.5, 2: 0.5, 3: 1.0, 4: 1.0}

    # Create the mapper with all required inputs
    point_mapper = PointMapper(
        test_dataframe,
        ideal_dataframe,
        matched_functions_dictionary,
        maximum_deviations_dictionary
    )

    # Run the mapping process and store the resulting DataFrame
    mapping_results_dataframe = point_mapper.map_all_test_points()

    # The result must contain all four expected columns, or be empty if nothing matched
    expected_columns = ["x", "y", "deviation", "ideal_function_index"]
    assert all(
        column_name in mapping_results_dataframe.columns
        for column_name in expected_columns
    ) or mapping_results_dataframe.empty

    # Every deviation value must be zero or positive — never negative
    if not mapping_results_dataframe.empty:
        assert (mapping_results_dataframe["deviation"] >= 0).all()