import pytest
import pandas as pd
import numpy as np
from base_classes import CSVDataHandler
from ideal_function_selector import FunctionSelector
from mapping import Test_Mapper
from db_manager import DatabaseManager


def test_csv_handler_missing_file(tmp_path):
    handler = CSVDataHandler(str(tmp_path / 'nope.csv'))
    with pytest.raises(Exception):
        handler.load_data()


def test_selector_invalid_lengths():
    # mismatched lengths should raise when comparing
    train = pd.DataFrame({'x': [1, 2, 3], 'y1': [1, 2, 3]})
    ideal = pd.DataFrame({'x': [1, 2], 'y1': [1, 2]})
    selector = FunctionSelector(train, ideal)
    with pytest.raises(Exception):
        selector.find_best_ideal_function(1)


def test_mapper_no_mappings():
    # Test points far away -> zero mappings
    x = np.linspace(0, 1, 5)
    ideal = pd.DataFrame({'x': x, 'y1': x, 'y2': 2*x, 'y3': 3*x, 'y4': 4*x})
    test_df = pd.DataFrame({'x': [100, 200], 'y': [1000, 2000]})
    selected = {1:1,2:2,3:3,4:4}
    max_devs = {1:0.1,2:0.1,3:0.1,4:0.1}
    mapper = Test_Mapper(test_df, ideal, selected, max_devs)
    mappings = mapper.map_all_test_points()
    assert mappings.empty


def test_db_insert_and_query(tmp_path):
    # create small db and insert then query
    db_path = str(tmp_path / 'test.db')
    db = DatabaseManager(db_name=db_path)
    db.create_tables()
    df_train = pd.DataFrame({'x':[0.0],'y1':[0.0],'y2':[0.0],'y3':[0.0],'y4':[0.0]})
    db.insert_training_data(df_train)
    got = db.get_training_data()
    assert len(got) == 1
    db.close()
