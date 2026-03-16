from sqlalchemy import create_engine, Column, Float, Integer, String, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from exceptions import DatabaseError

Base = declarative_base()

class TrainingData(Base):
	__tablename__ = 'training_data'

	id = Column(Integer, primary_key=True, autoincrement=True)
	x = Column(Float, nullable=False)
	y1 = Column(Float, nullable=False)
	y2 = Column(Float, nullable=False)
	y3 = Column(Float, nullable=False)
	y4 = Column(Float, nullable=False)

	def __repr__(self):
		return f"<TrainingData(x={self.x}, y1={self.y1})>"


class IdealFunction(Base):
	__tablename__ = 'ideal_functions'

	id = Column(Integer, primary_key=True, autoincrement=True)
	x = Column(Float, nullable=False)

	def __repr__(self):
		return f"<IdealFunction(x={self.x})>"


class TestMapping(Base):
	__tablename__ = 'test_mappings'

	id = Column(Integer, primary_key=True, autoincrement=True)
	x = Column(Float, nullable=False)
	y = Column(Float, nullable=False)
	deviation = Column(Float, nullable=False)
	ideal_function_index = Column(Integer, nullable=False)

	def __repr__(self):
		return f"<TestMapping(x={self.x}, y={self.y}, func={self.ideal_function_index})>"


class DatabaseManager:
	def __init__(self, db_name='assignment.db'):
		try:
			self.engine = create_engine(f'sqlite:///{db_name}', echo=False)
			Session = sessionmaker(bind=self.engine)
			self.session = Session()
			self.metadata = MetaData()
			print(f"Database connection established: {db_name}")
		except Exception as e:
			raise DatabaseError(f"Failed to create database: {str(e)}")

	def create_tables(self):
		try:
			Base.metadata.create_all(self.engine)
			print("Database tables created successfully")
		except Exception as e:
			raise DatabaseError(f"Failed to create tables: {str(e)}")

	def insert_training_data(self, df):
		try:
			df.to_sql('training_data', self.engine, if_exists='replace', index=False)
			print(f"Inserted {len(df)} rows into training_data table")
		except Exception as e:
			raise DatabaseError(f"Failed to insert training data: {str(e)}")

	def insert_ideal_functions(self, df):
		try:
			df.to_sql('ideal_functions', self.engine, if_exists='replace', index=False)
			print(f"Inserted {len(df)} rows into ideal_functions table")
		except Exception as e:
			raise DatabaseError(f"Failed to insert ideal functions: {str(e)}")

	def insert_test_mapping(self, x, y, deviation, ideal_function_index):
		try:
			mapping = TestMapping(x=x, y=y, deviation=deviation, ideal_function_index=ideal_function_index)
			self.session.add(mapping)
			self.session.commit()
		except Exception as e:
			self.session.rollback()
			raise DatabaseError(f"Failed to insert test mapping: {str(e)}")

	def insert_test_mappings_bulk(self, mappings_df):
		try:
			self.session.query(TestMapping).delete()
			self.session.commit()
			mappings_df.to_sql('test_mappings', self.engine, if_exists='append', index=False)
			print(f"Inserted {len(mappings_df)} test mappings")
		except Exception as e:
			self.session.rollback()
			raise DatabaseError(f"Failed to insert test mappings: {str(e)}")

	def get_training_data(self):
		try:
			query = "SELECT * FROM training_data"
			df = pd.read_sql(query, self.engine)
			return df
		except Exception as e:
			raise DatabaseError(f"Failed to retrieve training data: {str(e)}")

	def get_ideal_functions(self):
		try:
			query = "SELECT * FROM ideal_functions"
			df = pd.read_sql(query, self.engine)
			return df
		except Exception as e:
			raise DatabaseError(f"Failed to retrieve ideal functions: {str(e)}")

	def get_test_mappings(self):
		try:
			query = "SELECT * FROM test_mappings"
			df = pd.read_sql(query, self.engine)
			return df
		except Exception as e:
			raise DatabaseError(f"Failed to retrieve test mappings: {str(e)}")

	def close(self):
		self.session.close()
		print("Database connection closed")

