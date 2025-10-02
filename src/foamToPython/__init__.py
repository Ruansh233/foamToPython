"""foamToPython package"""

__all__ = [
    "readList",
    "readListList",
    "writeList",
    "OFField",
]

from .readOFList import readList, readListList, writeList
from .readOFField import OFField
