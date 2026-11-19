"""One versioned taxonomy for training histories and the demo catalog."""
import json
from pathlib import Path


def load_taxonomy(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if value.get("version") != 1 or "autre" not in value.get("categories", {}):
        raise ValueError("Expected common taxonomy version 1 with 'autre' fallback")
    for source in ("foursquare", "paris_data"):
        mapping = value.get(source)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Missing {source} category mapping")
        if any(not isinstance(k, str) or v not in value["categories"] for k, v in mapping.items()):
            raise ValueError(f"Invalid category in {source} mapping")
    return value


def paris_categories(tags, taxonomy):
    return sorted({taxonomy["paris_data"][t] for t in tags if t in taxonomy["paris_data"]}) or ["autre"]
