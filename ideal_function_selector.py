import numpy as np
import pandas as pd
from exception_classes import FunctionSelectionError


class IdealFunctionMatcher:
    """
    A class to match training functions to ideal functions using the
    Least Squares Method (LSM).


    """

    def __init__(self, training_dataframe, ideal_dataframe):
        """
        Initializes the IdealFunctionMatcher with training and ideal datasets.

        """
        self.training_dataframe = training_dataframe
        self.ideal_dataframe = ideal_dataframe

        # Stores the best matched ideal function index for each training function
        self.matches = {}

        # Stores the maximum deviation between each training function and its match
        self.deviation_map = {}

    def compute_sse(self, actual_values, predicted_values):
        """
        Computes the Sum of Squared Errors (SSE) between actual and predicted values.


        """
        try:
            # Ensure both arrays are the same length before computing SSE
            if len(actual_values) != len(predicted_values):
                raise FunctionSelectionError("Length mismatch between training and ideal data")

            # Compute element-wise residuals then square and sum them
            residuals = actual_values - predicted_values
            return np.sum(residuals ** 2)

        except FunctionSelectionError as error:
            print(f"[SSE Error] {error}")
            return None  # Return None to signal failure to the caller

    def find_optimal_match(self, training_function_index):
        """
        Finds the ideal function that best fits a given training function
        by minimizing the SSE across all 50 ideal functions.

        Args:
            training_function_index (int): Index of the training function (1 to 4).

        """
        try:
            training_column_name = f'y{training_function_index}'

            # Verify the requested training column exists in the dataset
            if training_column_name not in self.training_dataframe.columns:
                raise FunctionSelectionError(
                    f"Column {training_column_name} missing from training data"
                )

            actual_y_values = self.training_dataframe[training_column_name].values

            # Initialize tracking variables for the best match
            optimal_ideal_index = None
            lowest_sse_value = float('inf')   # Start with infinity so any SSE will be lower
            peak_deviation_value = None

            # Iterate over all 50 ideal functions to find the best match
            for ideal_column_number in range(1, 51):
                candidate_column_name = f'y{ideal_column_number}'

                # Skip if this ideal column does not exist in the dataset
                if candidate_column_name not in self.ideal_dataframe.columns:
                    continue

                candidate_y_values = self.ideal_dataframe[candidate_column_name].values

                # Compute SSE between training function and current ideal candidate
                computed_sse_value = self.compute_sse(actual_y_values, candidate_y_values)

                # Skip this candidate if SSE computation failed
                if computed_sse_value is None:
                    continue

                # Update best match if current SSE is lower than the previous best
                if computed_sse_value < lowest_sse_value:
                    lowest_sse_value = computed_sse_value
                    optimal_ideal_index = ideal_column_number

                    # Track the largest single-point deviation for this match
                    peak_deviation_value = np.max(np.abs(actual_y_values - candidate_y_values))

            # If no valid match was found across all ideal functions, raise an error
            if optimal_ideal_index is None:
                raise FunctionSelectionError(
                    f"No valid ideal function found for training func {training_function_index}"
                )

            return optimal_ideal_index, lowest_sse_value, peak_deviation_value

        except FunctionSelectionError as error:
            print(f"[Matching Error] {error}")
            return None, None, None  # Safe fallback tuple on failure

    def run_selection(self):
        """
        Runs the full matching process for all 4 training functions.
        Populates self.matches and self.deviation_map with results.

        """
        print("\n" + "=" * 60)
        print("IDEAL FUNCTION MATCHING VIA LEAST SQUARES")
        print("=" * 60)

        # Loop through all 4 training functions (y1 to y4)
        for training_function_index in range(1, 5):
            print(f"\nProcessing training column y{training_function_index}...")

            optimal_ideal_index, lowest_sse_value, peak_deviation_value = self.find_optimal_match(
                training_function_index
            )

            # Skip storing results if matching failed for this training function
            if optimal_ideal_index is None:
                print(f"  ✗ Skipping y{training_function_index} — no match found")
                continue

            # Store the match and its deviation for later use
            self.matches[training_function_index] = optimal_ideal_index
            self.deviation_map[training_function_index] = peak_deviation_value

            print(f"  ✓ Matched to: Ideal y{optimal_ideal_index}")
            print(f"    SSE Value:      {lowest_sse_value:.4f}")
            print(f"    Peak Deviation: {peak_deviation_value:.4f}")

        print("\n" + "=" * 60)
        print("MATCHING COMPLETE")
        print("=" * 60)
        print(f"Final matches: {self.matches}\n")

        return self.matches

    def get_matches(self):
        """
        Returns the dictionary of matched ideal functions.

        """
        try:
            # Guard against calling this before run_selection() has been executed
            if not self.matches:
                raise FunctionSelectionError(
                    "Matches not yet computed. Run run_selection() first."
                )
            return self.matches

        except FunctionSelectionError as error:
            print(f"[Access Error] {error}")
            return {}  # Return empty dict as safe fallback

    def get_deviations(self):
        """
        Returns the dictionary of maximum deviations per training function.


        """
        try:
            # Guard against calling this before run_selection() has been executed
            if not self.deviation_map:
                raise FunctionSelectionError(
                    "Deviations not yet computed. Run run_selection() first."
                )
            return self.deviation_map

        except FunctionSelectionError as error:
            print(f"[Access Error] {error}")
            return {}  # Return empty dict as safe fallback

    def extract_matched_ideal_data(self):
        """
        Extracts and returns a DataFrame containing only the matched ideal functions,
        with columns renamed to reflect which training function they correspond to.


        """
        try:
            # Ensure matching has been performed before attempting extraction
            if not self.matches:
                raise FunctionSelectionError(
                    "No matches available. Run run_selection() first."
                )

            # Build list of columns to extract: x plus all matched ideal function columns
            columns_to_keep = ['x'] + [f'y{ideal_index}' for ideal_index in self.matches.values()]
            extracted_dataframe = self.ideal_dataframe[columns_to_keep].copy()

            # Build a renaming map so columns reflect their training function origin
            column_rename_mapping = {'x': 'x'}
            for training_index, ideal_index in self.matches.items():
                column_rename_mapping[f'y{ideal_index}'] = f'matched_func_{training_index}'

            # Apply column renaming in place
            extracted_dataframe.rename(columns=column_rename_mapping, inplace=True)
            return extracted_dataframe

        except FunctionSelectionError as error:
            print(f"[Extraction Error] {error}")
            return pd.DataFrame()  # Return empty DataFrame as safe fallback