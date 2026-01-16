#!/usr/bin/env python3
"""
Block Documentation Generator

Generates markdown documentation for all blocks from code introspection.
Preserves manually-written content between marker comments.

Usage:
    # Generate all docs
    poetry run python scripts/generate_block_docs.py

    # Check mode for CI (exits 1 if stale)
    poetry run python scripts/generate_block_docs.py --check

    # Migrate existing docs (add markers, preserve content)
    poetry run python scripts/generate_block_docs.py --migrate

    # Verbose output
    poetry run python scripts/generate_block_docs.py -v
"""

import argparse
import inspect
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

logger = logging.getLogger(__name__)

# Default output directory relative to repo root
DEFAULT_OUTPUT_DIR = (
    Path(__file__).parent.parent.parent.parent / "docs" / "integrations"
)


@dataclass
class FieldDoc:
    """Documentation for a single input/output field."""

    name: str
    description: str
    type_str: str
    required: bool
    default: Any = None
    advanced: bool = False
    hidden: bool = False
    placeholder: str | None = None


@dataclass
class BlockDoc:
    """Documentation data extracted from a block."""

    id: str
    name: str
    class_name: str
    description: str
    categories: list[str]
    category_descriptions: dict[str, str]
    inputs: list[FieldDoc]
    outputs: list[FieldDoc]
    block_type: str
    source_file: str
    contributors: list[str] = field(default_factory=list)


# Category to human-readable name mapping
CATEGORY_DISPLAY_NAMES = {
    "AI": "AI and Language Models",
    "BASIC": "Basic Operations",
    "TEXT": "Text Processing",
    "SEARCH": "Search and Information Retrieval",
    "SOCIAL": "Social Media and Content",
    "DEVELOPER_TOOLS": "Developer Tools",
    "DATA": "Data Processing",
    "LOGIC": "Logic and Control Flow",
    "COMMUNICATION": "Communication",
    "INPUT": "Input/Output",
    "OUTPUT": "Input/Output",
    "MULTIMEDIA": "Media Generation",
    "PRODUCTIVITY": "Productivity",
    "HARDWARE": "Hardware",
    "AGENT": "Agent Integration",
    "CRM": "CRM Services",
    "SAFETY": "AI Safety",
    "ISSUE_TRACKING": "Issue Tracking",
    "MARKETING": "Marketing",
}

# Category to doc file mapping (for grouping related blocks)
CATEGORY_FILE_MAP = {
    "BASIC": "basic",
    "TEXT": "text",
    "AI": "llm",
    "SEARCH": "search",
    "DATA": "data",
    "LOGIC": "logic",
    "COMMUNICATION": "communication",
    "MULTIMEDIA": "multimedia",
    "PRODUCTIVITY": "productivity",
}


def class_name_to_display_name(class_name: str) -> str:
    """Convert BlockClassName to 'Block Class Name'."""
    # Remove 'Block' suffix
    name = class_name.replace("Block", "")
    # Insert space before capitals
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # Handle consecutive capitals (e.g., 'HTTPRequest' -> 'HTTP Request')
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    return name.strip()


def type_to_readable(type_schema: dict[str, Any] | Any) -> str:
    """Convert JSON schema type to human-readable string."""
    if not isinstance(type_schema, dict):
        return str(type_schema) if type_schema else "Any"

    if "anyOf" in type_schema:
        # Union type - show options
        any_of = type_schema["anyOf"]
        if not isinstance(any_of, list):
            return "Any"
        options = []
        for opt in any_of:
            if isinstance(opt, dict) and opt.get("type") == "null":
                continue
            options.append(type_to_readable(opt))
        if len(options) == 1:
            return options[0]
        return " | ".join(options)

    if "allOf" in type_schema:
        all_of = type_schema["allOf"]
        if not isinstance(all_of, list) or not all_of:
            return "Any"
        return type_to_readable(all_of[0])

    schema_type = type_schema.get("type")

    if schema_type == "array":
        items = type_schema.get("items", {})
        item_type = type_to_readable(items)
        return f"List[{item_type}]"

    if schema_type == "object":
        if "additionalProperties" in type_schema:
            value_type = type_to_readable(type_schema["additionalProperties"])
            return f"Dict[str, {value_type}]"
        # Check if it's a specific model
        title = type_schema.get("title", "Object")
        return title

    if schema_type == "string":
        if "enum" in type_schema:
            return " | ".join(f'"{v}"' for v in type_schema["enum"][:3])
        if "format" in type_schema:
            return f"str ({type_schema['format']})"
        return "str"

    if schema_type == "integer":
        return "int"

    if schema_type == "number":
        return "float"

    if schema_type == "boolean":
        return "bool"

    if schema_type == "null":
        return "None"

    # Fallback
    return type_schema.get("title", schema_type or "Any")


def safe_get(d: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict-like object."""
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def extract_block_doc(block_cls: type) -> BlockDoc:
    """Extract documentation data from a block class."""
    block = block_cls.create()

    # Get source file
    try:
        source_file = inspect.getfile(block_cls)
        # Make relative to blocks directory
        blocks_dir = Path(source_file).parent
        while blocks_dir.name != "blocks" and blocks_dir.parent != blocks_dir:
            blocks_dir = blocks_dir.parent
        source_file = str(Path(source_file).relative_to(blocks_dir.parent))
    except (TypeError, ValueError):
        source_file = "unknown"

    # Extract input fields
    input_schema = block.input_schema.jsonschema()
    input_properties = safe_get(input_schema, "properties", {})
    if not isinstance(input_properties, dict):
        input_properties = {}
    required_raw = safe_get(input_schema, "required", [])
    # Handle edge cases where required might not be a list
    if isinstance(required_raw, (list, set, tuple)):
        required_inputs = set(required_raw)
    else:
        required_inputs = set()

    inputs = []
    for field_name, field_schema in input_properties.items():
        if not isinstance(field_schema, dict):
            continue
        # Skip credentials fields in docs (they're auto-handled)
        if "credentials" in field_name.lower():
            continue

        inputs.append(
            FieldDoc(
                name=field_name,
                description=safe_get(field_schema, "description", ""),
                type_str=type_to_readable(field_schema),
                required=field_name in required_inputs,
                default=safe_get(field_schema, "default"),
                advanced=safe_get(field_schema, "advanced", False) or False,
                hidden=safe_get(field_schema, "hidden", False) or False,
                placeholder=safe_get(field_schema, "placeholder"),
            )
        )

    # Extract output fields
    output_schema = block.output_schema.jsonschema()
    output_properties = safe_get(output_schema, "properties", {})
    if not isinstance(output_properties, dict):
        output_properties = {}

    outputs = []
    for field_name, field_schema in output_properties.items():
        if not isinstance(field_schema, dict):
            continue
        outputs.append(
            FieldDoc(
                name=field_name,
                description=safe_get(field_schema, "description", ""),
                type_str=type_to_readable(field_schema),
                required=True,  # Outputs are always produced
                hidden=safe_get(field_schema, "hidden", False) or False,
            )
        )

    # Get category info (sort for deterministic ordering since it's a set)
    categories = []
    category_descriptions = {}
    for cat in sorted(block.categories, key=lambda c: c.name):
        categories.append(cat.name)
        category_descriptions[cat.name] = cat.value

    # Get contributors
    contributors = []
    for contrib in block.contributors:
        contributors.append(contrib.name if hasattr(contrib, "name") else str(contrib))

    return BlockDoc(
        id=block.id,
        name=class_name_to_display_name(block.name),
        class_name=block.name,
        description=block.description,
        categories=categories,
        category_descriptions=category_descriptions,
        inputs=inputs,
        outputs=outputs,
        block_type=block.block_type.value,
        source_file=source_file,
        contributors=contributors,
    )


def generate_anchor(name: str) -> str:
    """Generate markdown anchor from block name."""
    return name.lower().replace(" ", "-").replace("(", "").replace(")", "")


def extract_manual_content(existing_content: str) -> dict[str, str]:
    """Extract content between MANUAL markers from existing file."""
    manual_sections = {}

    # Pattern: <!-- MANUAL: section_name -->content<!-- END MANUAL -->
    pattern = r"<!-- MANUAL: (\w+) -->\s*(.*?)\s*<!-- END MANUAL -->"
    matches = re.findall(pattern, existing_content, re.DOTALL)

    for section_name, content in matches:
        manual_sections[section_name] = content.strip()

    return manual_sections


def strip_markers(content: str) -> str:
    """Remove MANUAL markers from content."""
    # Remove opening markers
    content = re.sub(r"<!-- MANUAL: \w+ -->\s*", "", content)
    # Remove closing markers
    content = re.sub(r"\s*<!-- END MANUAL -->", "", content)
    return content.strip()


def extract_legacy_content(existing_content: str) -> dict[str, str]:
    """Extract content from legacy docs without markers (for migration)."""
    manual_sections = {}

    # Try to extract "How it works" section
    how_it_works_match = re.search(
        r"### How it works\s*\n(.*?)(?=\n### |\n## |\Z)", existing_content, re.DOTALL
    )
    if how_it_works_match:
        content = strip_markers(how_it_works_match.group(1).strip())
        if content and not content.startswith("|"):  # Not a table
            manual_sections["how_it_works"] = content

    # Try to extract "Possible use case" section
    use_case_match = re.search(
        r"### Possible use case\s*\n(.*?)(?=\n### |\n## |\n---|\Z)",
        existing_content,
        re.DOTALL,
    )
    if use_case_match:
        content = strip_markers(use_case_match.group(1).strip())
        if content:
            manual_sections["use_case"] = content

    return manual_sections


def generate_block_markdown(
    block: BlockDoc,
    manual_content: dict[str, str] | None = None,
    is_first_in_file: bool = True,
) -> str:
    """Generate markdown documentation for a single block."""
    manual_content = manual_content or {}
    lines = []

    # Block heading
    heading_level = "#" if is_first_in_file else "##"
    lines.append(f"{heading_level} {block.name}")
    lines.append("")

    # What it is (full description)
    lines.append("### What it is")
    lines.append(block.description or "No description available.")
    lines.append("")

    # How it works (manual section)
    lines.append("### How it works")
    how_it_works = manual_content.get(
        "how_it_works", "_Add technical explanation here._"
    )
    lines.append("<!-- MANUAL: how_it_works -->")
    lines.append(how_it_works)
    lines.append("<!-- END MANUAL -->")
    lines.append("")

    # Inputs table (auto-generated)
    visible_inputs = [f for f in block.inputs if not f.hidden]
    if visible_inputs:
        lines.append("### Inputs")
        lines.append("| Input | Description | Type | Required |")
        lines.append("|-------|-------------|------|----------|")
        for inp in visible_inputs:
            required = "Yes" if inp.required else "No"
            desc = inp.description or "-"
            # Normalize newlines and escape pipes for valid table syntax
            desc = desc.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {inp.name} | {desc} | {inp.type_str} | {required} |")
        lines.append("")

    # Outputs table (auto-generated)
    visible_outputs = [f for f in block.outputs if not f.hidden]
    if visible_outputs:
        lines.append("### Outputs")
        lines.append("| Output | Description | Type |")
        lines.append("|--------|-------------|------|")
        for out in visible_outputs:
            desc = out.description or "-"
            # Normalize newlines and escape pipes for valid table syntax
            desc = desc.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {out.name} | {desc} | {out.type_str} |")
        lines.append("")

    # Possible use case (manual section)
    lines.append("### Possible use case")
    use_case = manual_content.get("use_case", "_Add practical use case examples here._")
    lines.append("<!-- MANUAL: use_case -->")
    lines.append(use_case)
    lines.append("<!-- END MANUAL -->")
    lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def get_block_file_mapping(blocks: list[BlockDoc]) -> dict[str, list[BlockDoc]]:
    """
    Map blocks to their documentation files.

    Returns dict of {relative_file_path: [blocks]}
    """
    file_mapping = defaultdict(list)

    for block in blocks:
        # Determine file path based on source file or category
        source_path = Path(block.source_file)

        # If source is in a subdirectory (e.g., google/gmail.py), use that structure
        if len(source_path.parts) > 2:  # blocks/subdir/file.py
            subdir = source_path.parts[1]  # e.g., "google"
            # Use the Python filename as the md filename
            md_file = source_path.stem + ".md"  # e.g., "gmail.md"
            file_path = f"{subdir}/{md_file}"
        else:
            # Use category-based grouping for top-level blocks
            primary_category = block.categories[0] if block.categories else "BASIC"
            file_name = CATEGORY_FILE_MAP.get(primary_category, "misc")
            file_path = f"{file_name}.md"

        file_mapping[file_path].append(block)

    return dict(file_mapping)


def generate_overview_table(blocks: list[BlockDoc]) -> str:
    """Generate the overview table markdown (blocks.md)."""
    lines = []

    lines.append("# AutoGPT Blocks Overview")
    lines.append("")
    lines.append(
        'AutoGPT uses a modular approach with various "blocks" to handle different tasks. These blocks are the building blocks of AutoGPT workflows, allowing users to create complex automations by combining simple, specialized components.'
    )
    lines.append("")
    lines.append('!!! info "Creating Your Own Blocks"')
    lines.append("    Want to create your own custom blocks? Check out our guides:")
    lines.append("    ")
    lines.append(
        "    - [Build your own Blocks](https://docs.agpt.co/platform/new_blocks/) - Step-by-step tutorial with examples"
    )
    lines.append(
        "    - [Block SDK Guide](https://docs.agpt.co/platform/block-sdk-guide/) - Advanced SDK patterns with OAuth, webhooks, and provider configuration"
    )
    lines.append("")
    lines.append(
        "Below is a comprehensive list of all available blocks, categorized by their primary function. Click on any block name to view its detailed documentation."
    )
    lines.append("")

    # Group blocks by category
    by_category = defaultdict(list)
    for block in blocks:
        primary_cat = block.categories[0] if block.categories else "BASIC"
        by_category[primary_cat].append(block)

    # Sort categories
    category_order = [
        "BASIC",
        "DATA",
        "TEXT",
        "AI",
        "SEARCH",
        "SOCIAL",
        "COMMUNICATION",
        "DEVELOPER_TOOLS",
        "MULTIMEDIA",
        "PRODUCTIVITY",
        "LOGIC",
        "INPUT",
        "OUTPUT",
        "AGENT",
        "CRM",
        "SAFETY",
        "ISSUE_TRACKING",
        "HARDWARE",
        "MARKETING",
    ]

    for category in category_order:
        if category not in by_category:
            continue

        cat_blocks = sorted(by_category[category], key=lambda b: b.name)
        display_name = CATEGORY_DISPLAY_NAMES.get(category, category)

        lines.append(f"## {display_name}")
        lines.append("| Block Name | Description |")
        lines.append("|------------|-------------|")

        for block in cat_blocks:
            # Determine link path
            file_mapping = get_block_file_mapping([block])
            file_path = list(file_mapping.keys())[0]
            anchor = generate_anchor(block.name)

            # Short description (first sentence)
            short_desc = (
                block.description.split(".")[0]
                if block.description
                else "No description"
            )
            short_desc = short_desc.replace("\n", " ").replace("|", "\\|")

            lines.append(f"| [{block.name}]({file_path}#{anchor}) | {short_desc} |")

        lines.append("")

    return "\n".join(lines)


def load_all_blocks_for_docs() -> list[BlockDoc]:
    """Load all blocks and extract documentation."""
    from backend.blocks import load_all_blocks

    block_classes = load_all_blocks()
    blocks = []

    for _block_id, block_cls in block_classes.items():
        try:
            block_doc = extract_block_doc(block_cls)
            blocks.append(block_doc)
        except Exception as e:
            logger.warning(f"Failed to extract docs for {block_cls.__name__}: {e}")

    return blocks


def write_block_docs(
    output_dir: Path,
    blocks: list[BlockDoc],
    migrate: bool = False,
    verbose: bool = False,
) -> dict[str, str]:
    """
    Write block documentation files.

    Returns dict of {file_path: content} for all generated files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_mapping = get_block_file_mapping(blocks)
    generated_files = {}

    for file_path, file_blocks in file_mapping.items():
        full_path = output_dir / file_path

        # Create subdirectories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing content for manual section preservation
        existing_content = ""
        if full_path.exists():
            existing_content = full_path.read_text()

        # Generate content for each block
        content_parts = []
        for i, block in enumerate(sorted(file_blocks, key=lambda b: b.name)):
            # Try to extract manual content
            if migrate:
                manual_content = extract_legacy_content(existing_content)
            else:
                # Extract manual content specific to this block
                # Look for content after the block heading
                block_pattern = (
                    rf"(?:^|\n)##? {re.escape(block.name)}\s*\n(.*?)(?=\n##? |\Z)"
                )
                block_match = re.search(block_pattern, existing_content, re.DOTALL)
                if block_match:
                    manual_content = extract_manual_content(block_match.group(1))
                else:
                    manual_content = {}

            content_parts.append(
                generate_block_markdown(
                    block,
                    manual_content,
                    is_first_in_file=(i == 0),
                )
            )

        full_content = "\n".join(content_parts)
        generated_files[str(file_path)] = full_content

        if verbose:
            print(f"  Writing {file_path} ({len(file_blocks)} blocks)")

        full_path.write_text(full_content)

    # Generate overview file
    overview_content = generate_overview_table(blocks)
    overview_path = output_dir / "README.md"
    generated_files["README.md"] = overview_content
    overview_path.write_text(overview_content)

    if verbose:
        print("  Writing README.md (overview)")

    return generated_files


def check_docs_in_sync(output_dir: Path, blocks: list[BlockDoc]) -> bool:
    """
    Check if generated docs match existing docs.

    Returns True if in sync, False otherwise.
    """
    output_dir = Path(output_dir)
    file_mapping = get_block_file_mapping(blocks)

    all_match = True
    out_of_sync_details: list[tuple[str, list[str]]] = []

    for file_path, file_blocks in file_mapping.items():
        full_path = output_dir / file_path

        if not full_path.exists():
            block_names = [b.name for b in sorted(file_blocks, key=lambda b: b.name)]
            print(f"MISSING: {file_path}")
            print(f"  Blocks: {', '.join(block_names)}")
            out_of_sync_details.append((file_path, block_names))
            all_match = False
            continue

        existing_content = full_path.read_text()

        # Extract manual content from existing file
        manual_sections_by_block = {}
        for block in file_blocks:
            block_pattern = (
                rf"(?:^|\n)##? {re.escape(block.name)}\s*\n(.*?)(?=\n##? |\Z)"
            )
            block_match = re.search(block_pattern, existing_content, re.DOTALL)
            if block_match:
                manual_sections_by_block[block.name] = extract_manual_content(
                    block_match.group(1)
                )

        # Generate expected content and check each block individually
        content_parts = []
        mismatched_blocks = []
        for i, block in enumerate(sorted(file_blocks, key=lambda b: b.name)):
            manual_content = manual_sections_by_block.get(block.name, {})
            expected_block_content = generate_block_markdown(
                block,
                manual_content,
                is_first_in_file=(i == 0),
            )
            content_parts.append(expected_block_content)

            # Check if this specific block's section exists and matches
            block_pattern = (
                rf"(?:^|\n)(##? {re.escape(block.name)}\s*\n.*?)(?=\n##? |\Z)"
            )
            block_match = re.search(block_pattern, existing_content, re.DOTALL)
            if not block_match:
                mismatched_blocks.append(f"{block.name} (missing)")
            elif block_match.group(1).strip() != expected_block_content.strip():
                mismatched_blocks.append(block.name)

        expected_content = "\n".join(content_parts)

        if existing_content.strip() != expected_content.strip():
            print(f"OUT OF SYNC: {file_path}")
            if mismatched_blocks:
                print(f"  Affected blocks: {', '.join(mismatched_blocks)}")
            out_of_sync_details.append((file_path, mismatched_blocks))
            all_match = False

    # Check overview
    overview_path = output_dir / "README.md"
    if overview_path.exists():
        existing_overview = overview_path.read_text()
        expected_overview = generate_overview_table(blocks)
        if existing_overview.strip() != expected_overview.strip():
            print("OUT OF SYNC: README.md (overview)")
            print("  The blocks overview table needs regeneration")
            out_of_sync_details.append(("README.md", ["overview table"]))
            all_match = False
    else:
        print("MISSING: README.md (overview)")
        out_of_sync_details.append(("README.md", ["overview table"]))
        all_match = False

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Generate block documentation from code introspection"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated docs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if docs are in sync (for CI), exit 1 if not",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate existing docs (extract legacy manual content)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("Loading blocks...")
    blocks = load_all_blocks_for_docs()
    print(f"Found {len(blocks)} blocks")

    if args.check:
        print(f"Checking docs in {args.output_dir}...")
        in_sync = check_docs_in_sync(args.output_dir, blocks)
        if in_sync:
            print("All documentation is in sync!")
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("Documentation is out of sync!")
            print("=" * 60)
            print("\nTo fix this, run one of the following:")
            print("\n  Option 1 - Run locally:")
            print(
                "    cd autogpt_platform/backend && poetry run python scripts/generate_block_docs.py"
            )
            print("\n  Option 2 - Ask Claude Code to run it:")
            print('    "Run the block docs generator script to sync documentation"')
            print("\n" + "=" * 60)
            sys.exit(1)
    else:
        print(f"Generating docs to {args.output_dir}...")
        write_block_docs(
            args.output_dir,
            blocks,
            migrate=args.migrate,
            verbose=args.verbose,
        )
        print("Done!")


if __name__ == "__main__":
    main()
