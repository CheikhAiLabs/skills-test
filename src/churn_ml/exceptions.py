class DataValidationError(Exception):
    """Raised when raw data fails quality or schema checks."""


class ModelValidationError(Exception):
    """Raised when a trained model fails promotion gates."""
