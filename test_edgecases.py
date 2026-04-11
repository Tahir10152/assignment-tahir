import pytest
import pandas as pd
import numpy as np
from classes import CSVLoader
from ideal_function_selector import IdealFunctionMatcher
from mapping import PointMapper
from database import DBHandler


def test_csvloader_missing_file(tmp_path):
    """
    Tests that CSVLoader raises an exception when the file does not exist.
    The loader should not silently fail on a missing file.
    """
    # Build a path to a file that does not actually exist in the temp directory
    nonexistent_file_path = str(tmp_path / 'nope.csv')

    # Create a loader pointing to the missing file
    csv_loader_instance = CSVLoader(nonexistent_file_path)

    # Trying to load a file that does not exist should raise some kind of exception
    with pytest.raises(Exception):
        csv_loader_instance.load_data()


def test_idealfunctionmatcher_mismatched_lengths():
    """
    Tests that IdealFunctionMatcher raises an exception when the training
    and ideal datasets have different numbers of rows, since SSE cannot
    be calculated between arrays of different lengths.
    """
    # Create a training dataset with 3 rows
    training_dataframe = pd.DataFrame({
        'x':  [1, 2, 3],
        'y1': [1, 2, 3]
    })

    # Create an ideal dataset with only 2 rows — intentionally mismatched
    ideal_dataframe = pd.DataFrame({
        'x':  [1, 2],
        'y1': [1, 2]
    })

    # Create the matcher with the mismatched datasets
    function_matcher = IdealFunctionMatcher(training_dataframe, ideal_dataframe)

    # Finding the best match should fail because row counts do not align
    with pytest.raises(Exception):
        function_matcher.find_optimal_match(1)


def test_pointmapper_no_mappings_when_points_are_far():
    """
    Tests that PointMapper returns an empty DataFrame when all test points
    are so far from the ideal functions that none pass the deviation threshold.
    """
    # Create x values and build a simple ideal dataset with 4 functions
    x_values = np.linspace(0, 1, 5)
    ideal_dataframe = pd.DataFrame({
        'x':  x_values,
        'y1': x_values,
        'y2': 2 * x_values,
        'y3': 3 * x_values,
        'y4': 4 * x_values
    })

    # Create test points that are extremely far from all ideal functions
    # so that no point should pass the deviation threshold
    test_dataframe = pd.DataFrame({
        'x': [100, 200],
        'y': [1000, 2000]
    })

    # Map each training function to its corresponding ideal function index
    matched_functions_dictionary = {1: 1, 2: 2, 3: 3, 4: 4}

    # Set very tight deviation limits so nothing can possibly match
    maximum_deviations_dictionary = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}

    # Create the mapper with the far-away test points
    point_mapper = PointMapper(
        test_dataframe,
        ideal_dataframe,
        matched_functions_dictionary,
        maximum_deviations_dictionary
    )

    # Run the mapping — result should be completely empty
    mapping_results_dataframe = point_mapper.map_all_test_points()

    # No test point should have been mapped to any ideal function
    assert mapping_results_dataframe.empty


def test_dbhandler_insert_and_retrieve(tmp_path):
    """
    Tests that DBHandler can create tables, insert training data,
    and retrieve it back correctly from the database.
    """
    # Build a path for a temporary SQLite database file
    temporary_database_path = str(tmp_path / 'test.db')

    # Create the database handler pointing to the temp database
    database_handler = DBHandler(db_name=temporary_database_path)

    # Create all required tables in the database
    database_handler.create_tables()

    # Build a minimal training DataFrame with one row to insert
    training_dataframe = pd.DataFrame({
        'x':  [0.0],
        'y1': [0.0],
        'y2': [0.0],
        'y3': [0.0],
        'y4': [0.0]
    })

    # Insert the training data into the database
    database_handler.store_training_data(training_dataframe)

    # Retrieve the data back from the database
    retrieved_dataframe = database_handler.fetch_training_data()

    # The retrieved data should have exactly one row
    assert len(retrieved_dataframe) == 1

    # Close the database connection cleanly after the test
    database_handler.close_connection()