import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from exception_classes import DataMappingError


class PointMapper:
    """
    Handles mapping of test data points to their closest matched ideal functions.

    Each test point is checked against all selected ideal functions.
    A point is mapped only if its deviation from an ideal function stays
    within the allowed threshold (max_deviation x sqrt(2)).


    """

    def __init__(self, test_dataframe, ideal_dataframe, matched_functions, deviation_limits):
        """
        Sets up the PointMapper with all data needed for mapping.


        """
        self.test_dataframe = test_dataframe
        self.ideal_dataframe = ideal_dataframe
        self.matched_functions = matched_functions
        self.deviation_limits = deviation_limits

        # Stores successfully mapped points before converting to DataFrame
        self.results_list = []

        # Precompute sqrt(2) once so we don't recalculate it for every test point
        self.square_root_of_two = np.sqrt(2)

    def get_interpolated_y(self, x_coordinate, function_index):
        """
        Estimates the y-value of an ideal function at a given x using linear interpolation.
        This handles cases where the test x does not exactly match a known x in the ideal data.


        """
        # Pull out the x and y arrays for the chosen ideal function
        ideal_x_values = self.ideal_dataframe['x'].values
        ideal_y_values = self.ideal_dataframe[f'y{function_index}'].values

        # Build a linear interpolator — extrapolate if x_coordinate is outside the known range
        linear_interpolator = interp1d(
            ideal_x_values,
            ideal_y_values,
            kind='linear',
            fill_value='extrapolate'
        )

        # Run the interpolator and convert result to a plain float
        return float(linear_interpolator(x_coordinate))

    def find_best_match(self, x_coordinate, y_coordinate):
        """
        Finds the ideal function that best matches a single test point.

        Checks each selected ideal function and picks the one with the
        smallest deviation that still falls within the allowed threshold.


        """
        # Start with no match found
        best_function_index = None
        best_deviation = None

        # Start with infinity so that any real deviation will be smaller
        lowest_deviation_so_far = float('inf')

        # Check every selected ideal function against this test point
        for training_index, ideal_index in self.matched_functions.items():

            # Get the ideal y-value at this test x-coordinate via interpolation
            estimated_y_value = self.get_interpolated_y(x_coordinate, ideal_index)

            # Calculate how far the test point is from this ideal function
            current_deviation = abs(y_coordinate - estimated_y_value)

            # Compute the maximum allowed deviation for this training function
            allowed_threshold = self.deviation_limits[training_index] * self.square_root_of_two

            # Only consider this function if its deviation is within the allowed threshold
            if current_deviation <= allowed_threshold:

                # Among all valid matches, keep the one with the smallest deviation
                if current_deviation < lowest_deviation_so_far:
                    lowest_deviation_so_far = current_deviation
                    best_function_index = ideal_index
                    best_deviation = current_deviation

        return best_function_index, best_deviation

    def map_all_test_points(self):
        """
        Runs the mapping process for every test point in the dataset.

        For each test point, tries to find a matching ideal function.
        Points that don't pass the threshold check are counted as unmapped.


        """
        print("\n" + "=" * 60)
        print("MAPPING TEST POINTS TO IDEAL FUNCTIONS")
        print("=" * 60)
        print(f"Total test points to process: {len(self.test_dataframe)}")
        print(f"Threshold rule: deviation must be ≤ max_deviation × √2")
        print()

        # Counters to track how many points get successfully mapped vs skipped
        successfully_mapped_count = 0
        could_not_map_count = 0

        # Go through every row in the test dataset one at a time
        for row_index, row_data in self.test_dataframe.iterrows():
            x_coordinate = row_data['x']
            y_coordinate = row_data['y']

            # Try to find a matching ideal function for this test point
            matched_function_index, calculated_deviation = self.find_best_match(
                x_coordinate,
                y_coordinate
            )

            if matched_function_index is not None:
                # Save the successful mapping as a dictionary entry in the results list
                self.results_list.append({
                    'x': x_coordinate,
                    'y': y_coordinate,
                    'deviation': calculated_deviation,
                    'ideal_function_index': matched_function_index
                })
                successfully_mapped_count += 1
            else:
                # No ideal function was close enough for this test point
                could_not_map_count += 1

        print(f"Successfully mapped: {successfully_mapped_count} points")
        print(f"Could not map:       {could_not_map_count} points")
        print("=" * 60 + "\n")

        # Convert results list to a DataFrame, or return an empty one if nothing was mapped
        if self.results_list:
            return pd.DataFrame(self.results_list)
        else:
            return pd.DataFrame(columns=['x', 'y', 'deviation', 'ideal_function_index'])

    def get_mappings(self):
        """
        Returns all mapping results as a DataFrame.


        """
        # Guard against calling this before the mapping process has been run
        if not self.results_list:
            raise DataMappingError("No mappings yet. Run map_all_test_points() first.")
        return pd.DataFrame(self.results_list)

    def get_stats(self):
        """
        Computes and returns summary statistics about the mapping results.


        """
        total_number_of_test_points = len(self.test_dataframe)

        # If nothing was mapped yet, return a zeroed summary dictionary
        if not self.results_list:
            return {
                'total_test_points': total_number_of_test_points,
                'mapped_points':     0,
                'unmapped_points':   total_number_of_test_points,
                'mapping_rate':      0.0
            }

        results_dataframe = pd.DataFrame(self.results_list)
        number_of_mapped_points = len(results_dataframe)

        # Count how many test points were assigned to each ideal function
        mappings_per_function_counts = results_dataframe['ideal_function_index'].value_counts().to_dict()

        return {
            'total_test_points':     total_number_of_test_points,
            'mapped_points':         number_of_mapped_points,
            'unmapped_points':       total_number_of_test_points - number_of_mapped_points,
            'mapping_rate':          (number_of_mapped_points / total_number_of_test_points) * 100,
            'mappings_per_function': mappings_per_function_counts,
            'avg_deviation':         results_dataframe['deviation'].mean(),
            'max_deviation':         results_dataframe['deviation'].max(),
            'min_deviation':         results_dataframe['deviation'].min()
        }