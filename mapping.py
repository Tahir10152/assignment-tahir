import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from exceptions import DataMappingError

class Test_Mapper:
    def __init__(self, test_data, ideal_functions, selected_functions, max_deviations):
        self.test_data = test_data
        self.ideal_functions = ideal_functions
        self.selected_functions = selected_functions
        self.max_deviations = max_deviations
        self.mappings = []
        self.sqrt2 = np.sqrt(2)

    def interpolate_ideal_value(self, x_test, ideal_func_num):
        x_ideal = self.ideal_functions['x'].values
        y_ideal = self.ideal_functions[f'y{ideal_func_num}'].values

        interpolator = interp1d(
            x_ideal,
            y_ideal,
            kind='linear',
            fill_value='extrapolate'
        )

        y_interpolated = float(interpolator(x_test))
        return y_interpolated

    def find_matching_function(self, x_test, y_test):
        best_match = None
        best_deviation = None
        smallest_deviation = float('inf')

        for train_num, ideal_num in self.selected_functions.items():
            y_ideal = self.interpolate_ideal_value(x_test, ideal_num)
            deviation = abs(y_test - y_ideal)
            threshold = self.max_deviations[train_num] * self.sqrt2
            if deviation <= threshold:
                if deviation < smallest_deviation:
                    smallest_deviation = deviation
                    best_match = ideal_num
                    best_deviation = deviation

        return best_match, best_deviation

    def map_all_test_points(self):
        print("\n" + "=" * 60)
        print("MAPPING TEST DATA TO IDEAL FUNCTIONS")
        print("=" * 60)
        print(f"Total test points: {len(self.test_data)}")
        print(f"Mapping criterion: deviation ≤ max_deviation × √2")
        print()

        mapped_count = 0
        unmapped_count = 0

        for idx, row in self.test_data.iterrows():
            x_test = row['x']
            y_test = row['y']
            ideal_func, deviation = self.find_matching_function(x_test, y_test)

            if ideal_func is not None:
                self.mappings.append({
                    'x': x_test,
                    'y': y_test,
                    'delta_y': deviation,
                    'ideal_func_no': ideal_func
                })
                mapped_count += 1
            else:
                unmapped_count += 1

        print(f"Mapped points: {mapped_count}")
        print(f"Unmapped points: {unmapped_count}")
        print("=" * 60 + "\n")

        if self.mappings:
            mappings_df = pd.DataFrame(self.mappings)
            return mappings_df
        else:
            return pd.DataFrame(columns=['x', 'y', 'delta_y', 'ideal_func_no'])

    def get_mappings(self):
        if not self.mappings:
            raise DataMappingError("No mappings created yet. Call map_all_test_points() first.")
        return pd.DataFrame(self.mappings)

    def get_mapping_statistics(self):
        if not self.mappings:
            return {
                'total_test_points': len(self.test_data),
                'mapped_points': 0,
                'unmapped_points': len(self.test_data),
                'mapping_rate': 0.0
            }

        mappings_df = pd.DataFrame(self.mappings)
        total = len(self.test_data)
        mapped = len(mappings_df)

        mappings_per_func = mappings_df['ideal_func_no'].value_counts().to_dict()

        return {
            'total_test_points': total,
            'mapped_points': mapped,
            'unmapped_points': total - mapped,
            'mapping_rate': (mapped / total) * 100,
            'mappings_per_function': mappings_per_func,
            'avg_deviation': mappings_df['delta_y'].mean(),
            'max_deviation': mappings_df['delta_y'].max(),
            'min_deviation': mappings_df['delta_y'].min()
        }
