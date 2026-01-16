#!/usr/bin/env python3
"""
Migration script to preserve manual content from existing docs.

This script:
1. Reads all existing block documentation (from git HEAD)
2. Extracts manual content (How it works, Possible use case) by block name
3. Creates a JSON mapping of block_name -> manual_content
4. Generates new docs using current block structure while preserving manual content
"""

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_block_docs import (
    generate_block_markdown,
    generate_overview_table,
    get_block_file_mapping,
    load_all_blocks_for_docs,
    strip_markers,
)


def get_git_file_content(file_path: str) -> str | None:
    """Get file content from git HEAD."""
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{file_path}"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,  # repo root
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def extract_blocks_from_doc(content: str) -> dict[str, dict[str, str]]:
    """Extract all block sections and their manual content from a doc file."""
    blocks = {}

    # Find all block headings (# or ##)
    block_pattern = r"(?:^|\n)(##?) ([^\n]+)\n"
    matches = list(re.finditer(block_pattern, content))

    for i, match in enumerate(matches):
        block_name = match.group(2).strip()
        start = match.end()

        # Find end (next heading or end of file)
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(content)

        block_content = content[start:end]

        # Extract manual sections
        manual_content = {}

        # How it works
        how_match = re.search(
            r"### How it works\s*\n(.*?)(?=\n### |\Z)", block_content, re.DOTALL
        )
        if how_match:
            text = strip_markers(how_match.group(1).strip())
            # Skip if it's just placeholder or a table
            if text and not text.startswith("|") and not text.startswith("_Add"):
                manual_content["how_it_works"] = text

        # Possible use case
        use_case_match = re.search(
            r"### Possible use case\s*\n(.*?)(?=\n### |\n## |\n---|\Z)",
            block_content,
            re.DOTALL,
        )
        if use_case_match:
            text = strip_markers(use_case_match.group(1).strip())
            if text and not text.startswith("_Add"):
                manual_content["use_case"] = text

        if manual_content:
            blocks[block_name] = manual_content

    return blocks


def collect_existing_manual_content() -> dict[str, dict[str, str]]:
    """Collect all manual content from existing git HEAD docs."""
    all_manual_content = {}

    # Find all existing md files via git
    result = subprocess.run(
        ["git", "ls-files", "docs/integrations/"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent.parent,
    )

    if result.returncode != 0:
        print("Failed to list git files")
        return {}

    for file_path in result.stdout.strip().split("\n"):
        if not file_path.endswith(".md"):
            continue
        if file_path.endswith("blocks.md"):  # Skip overview
            continue

        print(f"Processing: {file_path}")
        content = get_git_file_content(file_path)
        if content:
            blocks = extract_blocks_from_doc(content)
            for block_name, manual_content in blocks.items():
                if block_name in all_manual_content:
                    # Merge if already exists
                    all_manual_content[block_name].update(manual_content)
                else:
                    all_manual_content[block_name] = manual_content

    return all_manual_content


def run_migration():
    """Run the migration."""
    print("Step 1: Collecting existing manual content from git HEAD...")
    manual_content_cache = collect_existing_manual_content()

    print(f"\nFound manual content for {len(manual_content_cache)} blocks")

    # Show some examples
    for name, content in list(manual_content_cache.items())[:3]:
        print(f"  - {name}: {list(content.keys())}")

    # Save cache for reference
    cache_path = Path(__file__).parent / "manual_content_cache.json"
    with open(cache_path, "w") as f:
        json.dump(manual_content_cache, f, indent=2)
    print(f"\nSaved cache to {cache_path}")

    print("\nStep 2: Loading blocks from code...")
    blocks = load_all_blocks_for_docs()
    print(f"Found {len(blocks)} blocks")

    print("\nStep 3: Generating new documentation...")
    output_dir = Path(__file__).parent.parent.parent.parent / "docs" / "integrations"

    file_mapping = get_block_file_mapping(blocks)

    # Track statistics
    preserved_count = 0
    missing_count = 0

    for file_path, file_blocks in file_mapping.items():
        full_path = output_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        content_parts = []
        for i, block in enumerate(sorted(file_blocks, key=lambda b: b.name)):
            # Look up manual content by block name
            manual_content = manual_content_cache.get(block.name, {})

            if manual_content:
                preserved_count += 1
            else:
                # Try with class name
                manual_content = manual_content_cache.get(block.class_name, {})
                if manual_content:
                    preserved_count += 1
                else:
                    missing_count += 1

            content_parts.append(
                generate_block_markdown(
                    block,
                    manual_content,
                    is_first_in_file=(i == 0),
                )
            )

        full_content = "\n".join(content_parts)
        full_path.write_text(full_content)
        print(f"  Wrote {file_path} ({len(file_blocks)} blocks)")

    # Generate overview
    overview_content = generate_overview_table(blocks)
    overview_path = output_dir / "blocks.md"
    overview_path.write_text(overview_content)
    print("  Wrote blocks.md (overview)")

    print("\nMigration complete!")
    print(f"  - Blocks with preserved manual content: {preserved_count}")
    print(f"  - Blocks without manual content: {missing_count}")
    print(
        "\nYou can now run `poetry run python scripts/generate_block_docs.py --check` to verify"
    )


if __name__ == "__main__":
    run_migration()
