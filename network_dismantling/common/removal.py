"""Data structures for network dismantling removals.

This module defines the Removal dataclass representing a single node removal
during the dismantling process. All size values are ABSOLUTE counts, not fractions.
"""
from dataclasses import dataclass
from typing import List


@dataclass(slots=True, frozen=True)
class Removal:
    """A single node removal during network dismantling.
    
    All size values (lcc_size, slcc_size) are ABSOLUTE node counts, not fractions.
    To compute fractions at runtime: lcc_fraction = lcc_size / network_size
    
    Attributes:
        removal_num: Sequential removal index (1-indexed).
        node_id: Static ID of the removed node.
        prediction: Prediction value/score for this node.
        lcc_size: Absolute size of LCC after removal (node count).
        slcc_size: Absolute size of SLCC after removal (node count).
    """
    removal_num: int
    node_id: int
    prediction: float
    lcc_size: int      # Absolute count, NOT fraction
    slcc_size: int     # Absolute count, NOT fraction
    
    def to_tuple(self) -> tuple[int, int, float, int, int]:
        """Convert to tuple for backward compatibility."""
        return (self.removal_num, self.node_id, self.prediction, self.lcc_size, self.slcc_size)
    
    @classmethod
    def from_tuple(cls, t: tuple[int, int, float, int, int]) -> 'Removal':
        """Create from tuple for backward compatibility."""
        return cls(*t)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "removal_num": self.removal_num,
            "node_id": self.node_id,
            "prediction": self.prediction,
            "lcc_size": self.lcc_size,
            "slcc_size": self.slcc_size,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Removal':
        """Create from dictionary."""
        return cls(
            removal_num=d["removal_num"],
            node_id=d["node_id"],
            prediction=d["prediction"],
            lcc_size=d["lcc_size"],
            slcc_size=d["slcc_size"],
        )
    
    def __iter__(self):
        """Allow unpacking like a tuple."""
        return iter(self.to_tuple())
    
    def __getitem__(self, key):
        """Allow indexing like a tuple."""
        return self.to_tuple()[key]


# Type alias for list of removals
RemovalsList = List[Removal]


def removals_to_tuples(removals: RemovalsList) -> List[tuple[int, int, float, int, int]]:
    """Convert list of Removal objects to list of tuples."""
    return [r.to_tuple() for r in removals]


def removals_from_tuples(tuples: List[tuple[int, int, float, int, int]]) -> RemovalsList:
    """Convert list of tuples to list of Removal objects."""
    return [Removal.from_tuple(t) for t in tuples]
