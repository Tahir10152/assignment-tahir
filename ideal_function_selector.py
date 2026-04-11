import numpy as np
import pandas as pd
from exception_classes import FunctionSelectionError


class FunctionSelector:
	def __init__(self, training_data, ideal_functions):
		self.training_data = training_data
		self.ideal_functions = ideal_functions
		self.selected_functions = {}
		self.max_deviations = {}

	def calculate_least_squares(self, y_train, y_ideal):
		if len(y_train) != len(y_ideal):
			raise FunctionSelectionError("Training and ideal data must have same length")

		differences = y_train - y_ideal
		squared_differences = differences ** 2
		sum_squared = np.sum(squared_differences)
		return sum_squared

	def find_best_ideal_function(self, training_func_num):
		train_col = f'y{training_func_num}'
		if train_col not in self.training_data.columns:
			raise FunctionSelectionError(f"Training function {train_col} not found")

		y_train = self.training_data[train_col].values

		best_ideal_num = None
		min_sum_squares = float('inf')
		best_max_deviation = None

		for i in range(1, 51):
			ideal_col = f'y{i}'
			if ideal_col not in self.ideal_functions.columns:
				continue

			y_ideal = self.ideal_functions[ideal_col].values
			sum_squares = self.calculate_least_squares(y_train, y_ideal)

			if sum_squares < min_sum_squares:
				min_sum_squares = sum_squares
				best_ideal_num = i
				deviations = np.abs(y_train - y_ideal)
				best_max_deviation = np.max(deviations)

		if best_ideal_num is None:
			raise FunctionSelectionError("Could not find suitable ideal function")

		return best_ideal_num, min_sum_squares, best_max_deviation

	def select_all_functions(self):
		print("\n" + "=" * 60)
		print("SELECTING IDEAL FUNCTIONS (LEAST SQUARES METHOD)")
		print("=" * 60)

		for trainNumber in range(1, 5):
			print(f"\nAnalyzing Training Function {trainNumber}...")
			best_ideal, min_sum_sq, max_dev = self.find_best_ideal_function(trainNumber)
			self.selected_functions[trainNumber] = best_ideal
			self.max_deviations[trainNumber] = max_dev
			print(f"  ✓ Best match: Ideal Function {best_ideal}")
			print(f"    Sum of Squares: {min_sum_sq:.4f}")
			print(f"    Max Deviation: {max_dev:.4f}")

		print("\n" + "=" * 60)
		print("SELECTION COMPLETE")
		print("=" * 60)
		print(f"Selected functions: {self.selected_functions}")
		print()

		return self.selected_functions

	def get_selected_functions(self):
		if not self.selected_functions:
			raise FunctionSelectionError("No functions selected yet. Call select_all_functions() first.")
		return self.selected_functions

	def get_max_deviations(self):
		if not self.max_deviations:
			raise FunctionSelectionError("No deviations calculated yet. Call select_all_functions() first.")
		return self.max_deviations

	def get_selected_ideal_data(self):
		if not self.selected_functions:
			raise FunctionSelectionError("No functions selected yet.")

		selected_cols = ['x']
		for trainNumber, idealNumber in self.selected_functions.items():
			selected_cols.append(f'y{idealNumber}')

		selected_df = self.ideal_functions[selected_cols].copy()

		new_names = {'x': 'x'}
		for trainNumber, idealNumber in self.selected_functions.items():
			new_names[f'y{idealNumber}'] = f'ideal_func_{trainNumber}'

		selected_df.rename(columns=new_names, inplace=True)
		return selected_df

