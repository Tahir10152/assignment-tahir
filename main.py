import pandas as pd
from classes import CSVDataHandler
from ideal_function_selector import FunctionSelector
from mapping import TestMapper
from data_visualizer import Visualizer
from database import DatabaseManager


def run_pipeline():
	print('Starting simplified assignment pipeline')

	# Load datasets from repo root
	train = CSVDataHandler('train.csv')
	ideal = CSVDataHandler('ideal.csv')
	test = CSVDataHandler('test.csv')

	train.load_data(); ideal.load_data(); test.load_data()

	train.validate_data(['x', 'y1', 'y2', 'y3', 'y4'])
	ideal.validate_data(['x'])
	test.validate_data(['x', 'y'])

	train_df = train.data
	ideal_df = ideal.data
	test_df = test.data

	database_manager = DatabaseManager('assignment.db')
	database_manager.create_tables()
	database_manager.insert_training_data(train_df)
	database_manager.insert_ideal_functions(ideal_df)

	selector = FunctionSelector(train_df, ideal_df)
	selected = selector.select_all_functions()
	max_devs = selector.get_max_deviations()

	mapper = TestMapper(test_df, ideal_df, selected, max_devs)
	mappings = mapper.map_all_test_points()

	if not mappings.empty:
		database_manager.insert_test_mappings_bulk(mappings)

	viz = Visualizer()
	viz.create_all_visualizations(train_df, test_df, ideal_df, selected, mappings, max_devs)

	print('Pipeline finished')
	print('Pipeline finished')


if __name__ == '__main__':
	run_pipeline()

