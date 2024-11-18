# parameters.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import os

@dataclass
class StitchingParameters:
    """Parameters for image stitching operations."""
    input_folder: str
    output_format: str = '.ome.zarr'
    apply_flatfield: bool = False
    use_registration: bool = False
    registration_channel: str = ''
    registration_z_level: int = 0
    overlap_percent: float = 10.0
    flexible: bool = True
    coordinate_based: bool = True
    scan_pattern: str = 'Unidirectional' # 'S-Pattern'
    merge_timepoints: bool = True
    merge_hcs_regions: bool = False

    def __post_init__(self):        
        # Convert relative path to absolute
        self.input_folder = os.path.abspath(self.input_folder)
            
    def validate(self) -> None:
        """Validate parameters and raise appropriate errors."""
        if not os.path.exists(self.input_folder):
            raise ValueError(f"Input folder does not exist: {self.input_folder}")
            
        if self.use_registration:
            if not (0 < self.overlap_percent < 100):
                raise ValueError("Overlap percentage must be between 0 and 100")
            if not self.registration_channel:
                raise ValueError("Registration channel must be specified when using registration")
                
        if self.output_format not in ['.ome.zarr', '.ome.tiff']:
            raise ValueError("Output format must be either .ome.zarr or .ome.tiff")
    
    @property
    def stitched_folder(self) -> str:
        """Path to folder containing stitched outputs."""
        return os.path.join(self.input_folder + "_stitched")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StitchingParameters':
        """Create parameters from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_path: str) -> 'StitchingParameters':
        """Create parameters from a JSON file."""
        with open(json_path) as f:
            return cls.from_dict(json.load(f))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
    
    def to_json(self, json_path: str) -> None:
        """Save parameters to a JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
