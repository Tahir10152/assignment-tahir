import pandas as pd
from exceptions import DataValidationError, MissingColumnsError


class DataHandler:

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        print(f"Handler created for: {self.filepath}")

    def load_data(self):
        pass

    def validate_data(self):
        pass

    def get_data(self):
        try:
            if self.data is None:
                DataValidationError("Data has not been loaded yet.")
                return None
            print("Data retrieved successfully!")
            return self.data
        except Exception as e:
            print(f"Something went wrong while getting data: {e}")
            print("Tip: Make sure you call load_data() before get_data()")
            return None

    def get_column_names(self):
        try:
            if self.data is None:
                DataValidationError("Data has not been loaded yet.")
                return None
            list_of_column_names = list(self.data.columns)
            print(f"Columns in your data: {list_of_column_names}")
            return list_of_column_names
        except Exception as e:
            print(f"Something went wrong while getting column names: {e}")
            print("Tip: Make sure you call load_data() before get_column_names()")
            return None


class CSVDataHandler(DataHandler):

    def load_data(self):
        try:

            self.data = pd.read_csv(self.filepath)
            number_of_rows = len(self.data)
            print(f"Success! Loaded {number_of_rows} rows from {self.filepath}")
            return self.data
        except FileNotFoundError:
            print(f"Error: Could not find the file at '{self.filepath}'")
            print("Tip: Double check your file path and make sure the file exists")
        except Exception as e:
            print(f"Error: Something went wrong while loading the file")
            print(f"Details: {str(e)}")

    def validate_data(self, required_columns=None):
        try:
            print("Starting validation of your data...")

            # Step 1 - Make sure data is loaded first
            if self.data is None:
                DataValidationError("No data found! You need to load data first.")
                print("Tip: Call load_data() before validate_data()")
                return False

            # Step 2 - Check if all required columns exist
            if required_columns:
                print(f"Checking if these columns exist: {required_columns}")
                missing_cols = [
                    col for col in required_columns
                    if col not in self.data.columns
                ]
                if missing_cols:
                    MissingColumnsError(f"These columns are missing: {missing_cols}")
                    return False
                else:
                    print("All required columns are present!")

            # Step 3 - Check for empty/missing values
            data_has_missing_values = self.data.isnull().any().any()
            if data_has_missing_values:
                print("Warning: Your data has some empty/missing cells")
                print("Tip: You may want to fill or remove these before using the data")
            else:
                print("No missing values found in your data!")

            # Step 4 - All checks passed
            number_of_rows    = self.data.shape[0]
            number_of_columns = self.data.shape[1]
            print(f"Validation passed! Rows: {number_of_rows}, Columns: {number_of_columns}")
            return True

        except Exception as e:
            print(f"Error: Something went wrong during validation")
            print(f"Details: {str(e)}")
            return False