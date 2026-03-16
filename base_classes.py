import pandas as pd
from abc import ABC, abstractmethod
from exceptions import DataValidationError


class DataHandler(ABC):
	def __init__(self, filepath):
		self.filepath = filepath
		self.data = None

	@abstractmethod
	def load_data(self):
		pass

	@abstractmethod
	def validate_data(self):
		pass

	def get_data(self):
		if self.data is None:
			raise DataValidationError('Data has not been loaded yet.')

	def get_column_names(self):
		if self.data is None:
			raise DataValidationError('Data has not been loaded yet.')
		return list(self.data.columns)


class CSVDataHandler(DataHandler):
	def load_data(self):
		try:
			self.data = pd.read_csv(self.filepath)
			print(f"Successfully loaded {len(self.data)} rows from {self.filepath}")
			return self.data
		except FileNotFoundError:
			raise DataValidationError(f"File not found {self.filepath}")
		except Exception as e:
			raise DataValidationError(f"Error loading file {str(e)}")

	def validate_data(self, required_columns=None):
		if self.data is None:
			raise DataValidationError('No data to validate, load the correct data first.')
		if required_columns:
			missing_cols = [col for col in required_columns if col not in self.data.columns]
			if missing_cols:
				raise DataValidationError(f'Missing required columns: {missing_cols}')

		if self.data.isnull().any().any():
			print('Data contains missing values')

		print(f"Data validation passed - Shape: {self.data.shape}")
		return True

