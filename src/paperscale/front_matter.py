"""Front matter parsing utilities with minimal dependencies.

This module is intentionally kept lightweight (no numpy/torch/PIL) so it can be
imported from pipeline.py and other contexts that don't have GPU dependencies.
"""

import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Type, TypeAlias, Union, get_args, get_origin

import yaml

from paperscale.prompts import PageResponse

# Type alias for samples (same as in dataloader.py)
Sample: TypeAlias = Dict[str, Any]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FrontMatterParser:
    """Parses YAML front matter from markdown content.

    Can be used standalone or as a pipeline step in the dataloader.
    When used as a pipeline step, call it with a sample dict.
    """

    front_matter_class: Optional[Type] = None

    def _is_optional_str(self, field_type: Any) -> bool:
        """Check if a type is Optional[str]."""
        origin = get_origin(field_type)
        args = get_args(field_type)
        return origin is Union and type(None) in args and str in args

    def _extract_front_matter_and_text(self, markdown_content: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML front matter and text from markdown content."""
        if markdown_content.startswith("---\n"):
            try:
                # Find the closing --- delimiter
                end_index = markdown_content.find("\n---", 4)
                if end_index != -1:
                    front_matter_str = markdown_content[4:end_index]
                    text = markdown_content[end_index + 4 :].strip()

                    # Parse YAML
                    front_matter = yaml.safe_load(front_matter_str) or {}
                    return front_matter, text
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse YAML front matter: {e}")

        return {}, markdown_content.strip()

    def _parse_front_matter(self, front_matter_dict: Dict[str, Any], text: str) -> Any:
        """Parse front matter dictionary into dataclass instance if front_matter_class is specified."""
        if not self.front_matter_class:
            return front_matter_dict

        # Get field names and types from the dataclass
        field_info = {f.name: f.type for f in fields(self.front_matter_class)}

        # Validate and convert values
        kwargs: Dict[str, Any] = {}
        for field_name, field_type in field_info.items():
            # Special handling for natural_text field in PageResponse
            if field_name == "natural_text" and self.front_matter_class == PageResponse:
                kwargs[field_name] = text if text else None
                continue

            if field_name not in front_matter_dict:
                raise ValueError(f"Missing required field '{field_name}' in front matter")

            value = front_matter_dict[field_name]

            # Handle type conversions
            if field_type is int and isinstance(value, str):
                kwargs[field_name] = int(value)
            elif field_type is bool and isinstance(value, str):
                kwargs[field_name] = value.lower() == "true"
            elif self._is_optional_str(field_type):
                # Handle boolean values that YAML might produce (e.g., 'no' -> False)
                if isinstance(value, bool):
                    kwargs[field_name] = None
                elif isinstance(value, str):
                    kwargs[field_name] = value if value else None
                else:
                    kwargs[field_name] = None if not value else value
            else:
                kwargs[field_name] = value

        # Check for extra fields (excluding natural_text if it's PageResponse)
        expected_fields = set(field_info.keys())
        if self.front_matter_class == PageResponse:
            expected_fields.discard("natural_text")
        extra_fields = set(front_matter_dict.keys()) - expected_fields
        if extra_fields:
            raise ValueError(f"Unexpected fields in front matter: {extra_fields}")

        return self.front_matter_class(**kwargs)

    def __call__(self, sample: Sample) -> Sample:
        """Parse front matter from markdown content."""
        # Read markdown content if not already loaded
        if "markdown_content" not in sample:
            sample["markdown_content"] = sample["markdown_path"].read_text(encoding="utf-8")

        # Extract and parse front matter
        front_matter, text = self._extract_front_matter_and_text(sample["markdown_content"])

        # Parse front matter to dataclass if specified
        try:
            page_data = self._parse_front_matter(front_matter, text)
        except Exception as e:
            raise ValueError(f"Error parsing front matter for {sample['markdown_path']}: {e}")

        # Only add page_data field
        sample["page_data"] = page_data

        return sample
