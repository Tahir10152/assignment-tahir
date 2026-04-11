from sqlalchemy import create_engine, Column, Float, Integer, String, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from exception_classes import DatabaseError


# Base class for all SQLAlchemy ORM model definitions
Base = declarative_base()


class TrainRecord(Base):
    """
    ORM model representing a single row in the training_data table.


    """
    __tablename__ = 'training_data'

    # Primary key — auto-incremented by the database
    id = Column(Integer, primary_key=True, autoincrement=True)

    # x-coordinate shared across all training functions
    x = Column(Float, nullable=False)

    # y-values for each of the 4 training functions
    y1 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    y3 = Column(Float, nullable=False)
    y4 = Column(Float, nullable=False)

    def __repr__(self):
        return f"<TrainRecord(x={self.x}, y1={self.y1})>"


class IdealRecord(Base):
    """
    ORM model representing a single row in the ideal_functions table.
    The y-columns (y1–y50) are inserted dynamically via DataFrame,
    so only x is declared here as a fixed column.


    """
    __tablename__ = 'ideal_functions'

    # Primary key — auto-incremented by the database
    id = Column(Integer, primary_key=True, autoincrement=True)

    # x-coordinate shared across all 50 ideal functions
    x = Column(Float, nullable=False)

    def __repr__(self):
        return f"<IdealRecord(x={self.x})>"


class MappingRecord(Base):
    """
    ORM model representing a single mapped test point in the test_mappings table.


    """
    __tablename__ = 'test_mappings'

    # Primary key — auto-incremented by the database
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Coordinates of the test point
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)

    # How far this test point deviated from the matched ideal function
    deviation = Column(Float, nullable=False)

    # Which ideal function this test point was assigned to
    ideal_function_index = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<MappingRecord(x={self.x}, y={self.y}, ideal={self.ideal_function_index})>"


class DBHandler:
    """
    Manages all database operations for the assignment pipeline,
    including table creation, data insertion, and data retrieval.

    Uses SQLAlchemy for ORM-based table management and pandas for
    bulk DataFrame insertion via to_sql().


    """

    def __init__(self, db_name='assignment.db'):
        """
        Initializes the database engine and session.


        """
        try:
            # Create a SQLite engine — echo=False suppresses SQL query logging
            self.engine = create_engine(f'sqlite:///{db_name}', echo=False)

            # Create a session factory and open a session for ORM operations
            SessionFactory = sessionmaker(bind=self.engine)
            self.session = SessionFactory()

            # MetaData object used for schema reflection if needed
            self.metadata = MetaData()

            print(f"Database connection established: {db_name}")

        except Exception as e:
            raise DatabaseError(f"Failed to initialize database engine: {str(e)}")

    def create_tables(self):
        """
        Creates all ORM-defined tables in the database if they do not already exist.


        """
        try:
            # Reflect all Base subclass table definitions into the database
            Base.metadata.create_all(self.engine)
            print("Database tables created successfully")

        except Exception as e:
            raise DatabaseError(f"Failed to create tables: {str(e)}")

    def store_training_data(self, df):
        """
        Inserts the training DataFrame into the training_data table.
        Replaces the table if it already exists to avoid stale data.


        """
        try:
            # Replace existing table data on each run to keep the DB in sync
            df.to_sql('training_data', self.engine, if_exists='replace', index=False)
            print(f"Stored {len(df)} rows into training_data table")

        except Exception as e:
            raise DatabaseError(f"Failed to store training data: {str(e)}")

    def store_ideal_functions(self, df):
        """
        Inserts the ideal functions DataFrame into the ideal_functions table.
        Replaces the table if it already exists to avoid stale data.


        """
        try:
            # Replace existing table data on each run to keep the DB in sync
            df.to_sql('ideal_functions', self.engine, if_exists='replace', index=False)
            print(f"Stored {len(df)} rows into ideal_functions table")

        except Exception as e:
            raise DatabaseError(f"Failed to store ideal functions: {str(e)}")

    def store_single_mapping(self, x, y, deviation, ideal_func_idx):
        """
        Inserts a single test point mapping into the test_mappings table using ORM.


        """
        try:
            # Build an ORM record and add it to the current session
            record = MappingRecord(
                x=x,
                y=y,
                deviation=deviation,
                ideal_function_index=ideal_func_idx
            )
            self.session.add(record)
            self.session.commit()

        except Exception as e:
            # Roll back the session to undo any partial changes before raising
            self.session.rollback()
            raise DatabaseError(f"Failed to store single mapping: {str(e)}")

    def store_all_mappings(self, mappings_df):
        """
        Bulk-inserts all test point mappings into the test_mappings table.
        Replaces the table entirely to avoid schema mismatch errors when
        the on-disk table structure differs from the current DataFrame columns.


        """
        try:
            # Drop and recreate the table to ensure column schema stays in sync
            # with the current DataFrame structure (avoids "no column named" errors)
            mappings_df.to_sql('test_mappings', self.engine, if_exists='replace', index=False)
            print(f"Stored {len(mappings_df)} test mappings into test_mappings table")

        except Exception as e:
            # to_sql does not use the ORM session, so no rollback needed here
            raise DatabaseError(f"Failed to store bulk mappings: {str(e)}")

    def fetch_training_data(self):
        """
        Retrieves all rows from the training_data table.


        """
        try:
            return pd.read_sql("SELECT * FROM training_data", self.engine)

        except Exception as e:
            raise DatabaseError(f"Failed to fetch training data: {str(e)}")

    def fetch_ideal_functions(self):
        """
        Retrieves all rows from the ideal_functions table.


        """
        try:
            return pd.read_sql("SELECT * FROM ideal_functions", self.engine)

        except Exception as e:
            raise DatabaseError(f"Failed to fetch ideal functions: {str(e)}")

    def fetch_test_mappings(self):
        """
        Retrieves all rows from the test_mappings table.


        """
        try:
            return pd.read_sql("SELECT * FROM test_mappings", self.engine)

        except Exception as e:
            raise DatabaseError(f"Failed to fetch test mappings: {str(e)}")

    def close_connection(self):
        """
        Closes the active SQLAlchemy session to release database resources.
        Should be called at the end of the pipeline.
        """
        # Close the session to free up the connection pool
        self.session.close()
        print("Database connection closed")