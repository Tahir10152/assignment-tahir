class DataValidationError(Exception):
	"""Raised when input data is missing or invalid."""
	pass


class DatabaseError(Exception):
	"""Raised for database / persistence related errors."""
	pass


class FunctionSelectionError(Exception):
	"""Raised when matching/selecting ideal functions fails."""
	pass


class DataMappingError(Exception):
	"""Raised during test -> ideal mapping operations."""
	pass
