import pandas as pd
from exception_classes import DataValidationError, MissingColumnsError


class FileLoader:
    # This is the base class that holds common stuff for loading any kind of file
    # Other classes can build on top of this one

    def __init__(self, file_path):
        # Save the file path so we can use it later when loading
        self.file_path = file_path
        self.loaded_data = None
        print(f"File handler ready for: {self.file_path}")

    def load_data(self):
        # This is just a placeholder — the real loading happens in the child class
        pass

    def validate_data(self):
        # This is just a placeholder — the real checking happens in the child class
        pass

    def get_data(self):
        # This method gives back the data after it has been loaded
        try:
            # Check if data was loaded at all before trying to return it
            if self.loaded_data is None:
                DataValidationError("Looks like the data was never loaded!")
                return None
            print("Here is your data!")
            return self.loaded_data
        except Exception as e:
            print(f"Oops, something went wrong while getting the data: {e}")
            print("Hint: Make sure you ran load_data() before calling this")
            return None

    def get_column_names(self):
        # This method returns a list of all the column names in the loaded data
        try:
            # Can't get column names if nothing has been loaded yet
            if self.loaded_data is None:
                DataValidationError("Data is empty — nothing to get columns from!")
                return None
            all_columns = list(self.loaded_data.columns)
            print(f"Here are the column names: {all_columns}")
            return all_columns
        except Exception as e:
            print(f"Oops, something went wrong while getting column names: {e}")
            print("Hint: Make sure you ran load_data() before calling this")
            return None


class CSVLoader(FileLoader):
    # This class handles loading CSV files specifically
    # It extends FileLoader and fills in the actual logic

    def load_data(self):
        # Try to read the CSV file from the path we saved earlier
        try:
            self.loaded_data = pd.read_csv(self.file_path)
            total_rows = len(self.loaded_data)
            print(f"Done! Read {total_rows} rows from {self.file_path}")
            return self.loaded_data
        except FileNotFoundError:
            # The file just doesn't exist at that path
            print(f"Error: Can't find the file at '{self.file_path}'")
            print("Hint: Check the file path and make sure the file is actually there")
        except Exception as e:
            # Something else went wrong we didn't expect
            print(f"Error: Something went wrong while reading the file")
            print(f"More details: {str(e)}")

    def validate_data(self, required_columns=None):
        # This checks if the data looks good before we use it
        try:
            print("Checking your data now...")

            # Check 1 - we can't validate something that was never loaded
            if self.loaded_data is None:
                DataValidationError("There is no data to check! Load it first.")
                print("Hint: Run load_data() before validate_data()")
                return False

            # Check 2 - make sure all the columns we need are actually in the file
            if required_columns:
                print(f"Looking for these columns: {required_columns}")
                cols_not_found = [
                    col for col in required_columns
                    if col not in self.loaded_data.columns
                ]
                # If any columns are missing, stop and say which ones
                if cols_not_found:
                    MissingColumnsError(f"These columns are missing from the file: {cols_not_found}")
                    return False
                else:
                    print("Good news — all the columns you need are there!")

            # Check 3 - look for any empty or missing cells in the data
            has_empty_cells = self.loaded_data.isnull().any().any()
            if has_empty_cells:
                print("Heads up: Some cells in your data are empty or missing")
                print("Hint: You might want to fix those before doing anything with the data")
            else:
                print("All cells have values — no missing data found!")

            # Check 4 - everything passed, show a summary
            row_count = self.loaded_data.shape[0]
            col_count = self.loaded_data.shape[1]
            print(f"All checks passed! Your data has {row_count} rows and {col_count} columns")
            return True

        except Exception as e:
            print(f"Error: Something went wrong during the check")
            print(f"More details: {str(e)}")
            return False