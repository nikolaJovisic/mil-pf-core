from dataclasses import dataclass
from typing import List

from mil_pf_core.types.view_set import ViewSet

@dataclass
class ViewSetList:
    viewsets: List[ViewSet]
