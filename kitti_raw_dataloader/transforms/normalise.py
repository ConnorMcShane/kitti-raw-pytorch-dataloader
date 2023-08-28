"""Normalise class for transforms."""

from typing import Dict, Any


class Normalise():
    """Normalise class for transforms."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize the transform class.
        Args:
            params (dict): parameters for the transform
        """
        self.params: Dict[str, Any] = params

    def __call__(self, data: dict) -> dict:
        """Apply the transform to the data.
        Args:
            data (dict): data to apply the transform to

        Returns:
            data (dict): data with the transform applied
        """
        return self.apply_transform(data)

    def apply_transform(self, data: Dict) -> Dict:
        """Apply the transform to the data.
        Args:
            data (dict): data to apply the transform to

        Returns:
            data (dict): data with the transform applied
        """
        return data
