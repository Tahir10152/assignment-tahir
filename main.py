import pandas as pd
from classes import CSVLoader
from ideal_function_selector import IdealFunctionMatcher
from mapping import PointMapper
from data_visualizer import PlotEngine
from database import DBHandler
from header_footer_style import header_style

header_style("Best Function Selection and Test Point Mapping Pipeline")


def run_pipeline():
    """
    Main pipeline function that orchestrates the full workflow:
    - Loading and validating datasets
    - Storing data in the database
    - Matching training functions to ideal functions
    - Mapping test points to matched functions
    - Visualizing the results
    """
    # Pipeline introduction and student information
    print("Best Function Selection and Test Point Mapping Pipeline")
    print("Student: Mohammad Tahir Zaman")
    print("Course: CSEMDSPWP01 - Python Final Assignment")
    print("Starting simplified assignment pipeline")

    # ----------------------------------------------------------------
    # STEP 1: Load datasets from CSV files in the project root directory
    # ----------------------------------------------------------------
    header_style("Step 1: Load Datasets from CSV Files")

    training_csv_loader = CSVLoader('train.csv')
    ideal_csv_loader    = CSVLoader('ideal.csv')
    test_csv_loader     = CSVLoader('test.csv')

    # Trigger actual file reading for each dataset
    training_csv_loader.load_data()
    ideal_csv_loader.load_data()
    test_csv_loader.load_data()

    # ----------------------------------------------------------------
    # STEP 2: Validate that required columns exist in each dataset
    # ----------------------------------------------------------------
    header_style("Step 2: Validate Required Columns in Datasets")

    # Training data must have x and 4 y-columns (y1 to y4)
    training_csv_loader.validate_data(['x', 'y1', 'y2', 'y3', 'y4'])

    # Ideal data must at least have the x column (y-columns checked dynamically)
    ideal_csv_loader.validate_data(['x'])

    # Test data must have x and a single y column
    test_csv_loader.validate_data(['x', 'y'])

    # Extract raw DataFrames from the loaded_data attribute for further processing
    training_dataframe = training_csv_loader.loaded_data
    ideal_dataframe    = ideal_csv_loader.loaded_data
    test_dataframe     = test_csv_loader.loaded_data

    # ----------------------------------------------------------------
    # STEP 3: Initialize database and store raw datasets
    # ----------------------------------------------------------------

    header_style("Step 3: Initialize Database and Store Raw Datasets")

    database_handler = DBHandler('assignment.db')

    # Create all required tables if they don't already exist
    database_handler.create_tables()

    # Persist training and ideal function data into the database
    database_handler.store_training_data(training_dataframe)
    database_handler.store_ideal_functions(ideal_dataframe)

    # ----------------------------------------------------------------
    # STEP 4: Match each training function to its best ideal function
    # using the Least Squares Method
    # ----------------------------------------------------------------
    header_style("Step 4: Match Training Functions to Ideal Functions")

    function_matcher = IdealFunctionMatcher(training_dataframe, ideal_dataframe)

    # Run the full selection process across all 4 training functions
    selected_functions_dictionary = function_matcher.run_selection()

    # Retrieve the maximum deviations calculated during matching
    maximum_deviations_dictionary = function_matcher.get_deviations()

    # ----------------------------------------------------------------
    # STEP 5: Map each test point to one of the selected ideal functions
    # ----------------------------------------------------------------

    header_style("Step 5: Map Test Points to Selected Ideal Functions")

    test_point_mapper = PointMapper(
        test_dataframe,
        ideal_dataframe,
        selected_functions_dictionary,
        maximum_deviations_dictionary
    )

    # Run the mapping process and store the resulting DataFrame
    mapping_results_dataframe = test_point_mapper.map_all_test_points()

    # Only insert into the database if there are valid mappings to store
    if not mapping_results_dataframe.empty:
        database_handler.store_all_mappings(mapping_results_dataframe)

    # ----------------------------------------------------------------
    # STEP 6: Generate visualizations for training, test, and ideal data
    # ----------------------------------------------------------------
    header_style("Step 6: Generate Visualizations for Training, Test, and Ideal Data")

    plot_engine_instance = PlotEngine()
    plot_engine_instance.create_all_visualizations(
        training_dataframe,
        test_dataframe,
        ideal_dataframe,
        selected_functions_dictionary,
        mapping_results_dataframe,
        maximum_deviations_dictionary
    )

    print('Pipeline finished')


if __name__ == '__main__':
    run_pipeline()