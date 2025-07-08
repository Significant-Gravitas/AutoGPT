"""
Pytest-based webhook endpoint validation for all provider blocks.

This test automatically discovers all webhook trigger blocks and validates
their configurations, ensuring they properly define webhook endpoints and
event handling.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest


def extract_webhook_configurations(file_content: str) -> List[Dict]:
    """
    Extract webhook configurations from file content.
    Returns list of webhook configurations found.
    """
    configs = []

    # Pattern for BlockWebhookConfig - match until the closing paren at same or lower indent
    webhook_config_pattern = r"BlockWebhookConfig\s*\(((?:[^()]+|\([^()]*\))*)\)"
    matches = re.finditer(
        webhook_config_pattern, file_content, re.MULTILINE | re.DOTALL
    )

    for match in matches:
        config_str = match.group(1)
        config = {
            "type": "BlockWebhookConfig",
            "raw": config_str,
            "provider": None,
            "webhook_type": None,
            "resource_format": None,
            "event_filter_input": None,
            "event_format": None,
        }

        # Extract provider
        provider_match = re.search(
            r'provider\s*=\s*ProviderName\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)',
            config_str,
        )
        if not provider_match:
            provider_match = re.search(
                r"provider\s*=\s*ProviderName\.([A-Z_]+)", config_str
            )
        if not provider_match:
            # Try to match variable reference like ProviderName(generic_webhook.name)
            provider_match = re.search(
                r"provider\s*=\s*ProviderName\s*\(\s*(\w+)\.name\s*\)", config_str
            )
        if provider_match:
            provider_name = provider_match.group(1)
            config["provider"] = provider_name.lower()

        # Extract webhook_type
        webhook_type_match = re.search(
            r'webhook_type\s*=\s*["\']([\w_-]+)["\']', config_str
        )
        if not webhook_type_match:
            webhook_type_match = re.search(
                r"webhook_type\s*=\s*(\w+)\.(\w+)", config_str
            )
            if webhook_type_match:
                # Extract the enum value
                config["webhook_type"] = webhook_type_match.group(2).lower()
        if webhook_type_match and not config["webhook_type"]:
            config["webhook_type"] = webhook_type_match.group(1)

        # Extract resource_format
        resource_format_match = re.search(
            r'resource_format\s*=\s*["\']([^"\']*)["\']', config_str
        )
        if resource_format_match:
            config["resource_format"] = resource_format_match.group(1)

        # Extract event_filter_input
        event_filter_match = re.search(
            r'event_filter_input\s*=\s*["\']([^"\']+)["\']', config_str
        )
        if event_filter_match:
            config["event_filter_input"] = event_filter_match.group(1)

        # Extract event_format
        event_format_match = re.search(
            r'event_format\s*=\s*["\']([^"\']+)["\']', config_str
        )
        if event_format_match:
            config["event_format"] = event_format_match.group(1)

        configs.append(config)

    # Pattern for BlockManualWebhookConfig - match until the closing paren at same or lower indent
    manual_webhook_pattern = r"BlockManualWebhookConfig\s*\(((?:[^()]+|\([^()]*\))*)\)"
    matches = re.finditer(
        manual_webhook_pattern, file_content, re.MULTILINE | re.DOTALL
    )

    for match in matches:
        config_str = match.group(1)
        config = {
            "type": "BlockManualWebhookConfig",
            "raw": config_str,
            "provider": None,
            "webhook_type": None,
            "event_filter_input": None,
        }

        # Extract provider
        provider_match = re.search(
            r'provider\s*=\s*ProviderName\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)',
            config_str,
        )
        if not provider_match:
            provider_match = re.search(
                r"provider\s*=\s*ProviderName\.([A-Z_]+)", config_str
            )
        if not provider_match:
            # Try to match variable reference like ProviderName(generic_webhook.name)
            provider_match = re.search(
                r"provider\s*=\s*ProviderName\s*\(\s*(\w+)\.name\s*\)", config_str
            )
        if provider_match:
            provider_name = provider_match.group(1)
            config["provider"] = provider_name.lower()

        # Extract webhook_type
        webhook_type_match = re.search(
            r'webhook_type\s*=\s*["\']([\w_-]+)["\']', config_str
        )
        if not webhook_type_match:
            webhook_type_match = re.search(
                r"webhook_type\s*=\s*(\w+)\.(\w+)", config_str
            )
            if webhook_type_match:
                # Extract the enum value
                config["webhook_type"] = webhook_type_match.group(2).lower()
        if webhook_type_match and not config["webhook_type"]:
            config["webhook_type"] = webhook_type_match.group(1)

        # Extract event_filter_input
        event_filter_match = re.search(
            r'event_filter_input\s*=\s*["\']([^"\']+)["\']', config_str
        )
        if event_filter_match:
            config["event_filter_input"] = event_filter_match.group(1)

        configs.append(config)

    return configs


def extract_webhook_blocks(file_content: str) -> List[Tuple[str, int]]:
    """
    Extract webhook block class names and their line numbers.
    Returns list of (class_name, line_number) tuples.
    """
    blocks = []
    lines = file_content.split("\n")

    # Pattern for webhook block classes
    class_pattern = r"class\s+(\w+Block)\s*\(.*Block.*\):"

    for line_num, line in enumerate(lines, 1):
        match = re.search(class_pattern, line)
        if match:
            class_name = match.group(1)
            # Check if this is likely a webhook block by looking for BlockType.WEBHOOK
            # or webhook-related configurations in the next few lines
            is_webhook = False

            # Check next 20 lines for webhook indicators
            for i in range(line_num - 1, min(line_num + 19, len(lines))):
                if i < len(lines):
                    check_line = lines[i]
                    if (
                        "BlockType.WEBHOOK" in check_line
                        or "BlockWebhookConfig" in check_line
                        or "BlockManualWebhookConfig" in check_line
                        or "webhook_config=" in check_line
                    ):
                        is_webhook = True
                        break

            if is_webhook:
                blocks.append((class_name, line_num))

    return blocks


def get_all_webhook_files() -> Dict[str, List[Path]]:
    """Get all files that potentially contain webhook blocks."""
    test_dir = Path(__file__).parent
    backend_dir = test_dir.parent.parent
    blocks_dir = backend_dir / "backend" / "blocks"

    webhook_files = {}

    # Check all provider directories
    for provider_dir in blocks_dir.iterdir():
        if provider_dir.is_dir() and not provider_dir.name.startswith(("_", ".")):
            provider = provider_dir.name

            # Look for trigger files and webhook files
            trigger_files = list(provider_dir.glob("*trigger*.py"))
            webhook_files_list = list(provider_dir.glob("*webhook*.py"))

            # Combine and deduplicate
            all_files = list(set(trigger_files + webhook_files_list))

            if all_files:
                webhook_files[provider] = all_files

    return webhook_files


def load_webhook_spec(provider: str) -> Optional[Dict]:
    """Load webhook specification from JSON file."""
    spec_file = Path(__file__).parent / "test_data" / f"{provider}.json"

    if not spec_file.exists():
        return None

    with open(spec_file, "r") as f:
        spec = json.load(f)

    # Return webhook-specific configuration if it exists
    return spec.get("webhooks", {})


def validate_webhook_configuration(config: Dict, spec: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a webhook configuration against the specification.
    Returns (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields based on config type
    if config["type"] == "BlockWebhookConfig":
        required_fields = ["provider", "webhook_type"]
        for field in required_fields:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")

    # Validate against spec if available
    if spec:
        # Check if webhook_type is in allowed types
        allowed_types = spec.get("allowed_webhook_types", [])
        if allowed_types and config.get("webhook_type") not in allowed_types:
            errors.append(
                f"Invalid webhook_type '{config.get('webhook_type')}'. "
                f"Allowed types: {', '.join(allowed_types)}"
            )

        # Validate resource_format if specified
        if config.get("resource_format") is not None:
            expected_format = spec.get("resource_format_pattern")
            if expected_format and config["resource_format"] != expected_format:
                # Check if it's a valid pattern (contains placeholders)
                if (
                    not re.search(r"\{[^}]+\}", config["resource_format"])
                    and config["resource_format"]
                ):
                    errors.append(
                        f"Invalid resource_format '{config['resource_format']}'. "
                        f"Expected pattern like: {expected_format}"
                    )

    return len(errors) == 0, errors


@pytest.mark.parametrize(
    "provider",
    [
        "airtable",
        "baas",
        "elevenlabs",
        "exa",
        "github",
        "slant3d",
        "compass",
        "generic_webhook",
    ],
)
def test_provider_webhook_configurations(provider: str):
    """
    Test that all webhook configurations in provider implementations are valid.

    This test:
    1. Discovers all webhook blocks in the provider's code
    2. Extracts their webhook configurations
    3. Validates configurations have required fields
    4. Checks against specifications if available
    """
    webhook_files = get_all_webhook_files()

    if provider not in webhook_files:
        pytest.skip(f"No webhook files found for provider: {provider}")

    # Load webhook specification if available
    spec = load_webhook_spec(provider)

    # Extract all webhook configurations
    all_configs = []
    block_locations = {}  # block_name -> (file, line_num)

    for py_file in webhook_files[provider]:
        with open(py_file, "r") as f:
            content = f.read()

            # Extract webhook blocks
            blocks = extract_webhook_blocks(content)
            for block_name, line_num in blocks:
                block_locations[block_name] = (py_file.name, line_num)

            # Extract configurations
            configs = extract_webhook_configurations(content)
            for config in configs:
                config["file"] = py_file.name
                all_configs.append(config)

    # Validate all configurations
    validation_errors = []

    for config in all_configs:
        is_valid, errors = validate_webhook_configuration(config, spec or {})

        if not is_valid:
            error_msg = f"\n  ❌ Invalid webhook configuration in {config['file']}:"
            error_msg += f"\n     Type: {config['type']}"
            error_msg += f"\n     Provider: {config.get('provider', 'MISSING')}"
            error_msg += f"\n     Webhook Type: {config.get('webhook_type', 'MISSING')}"
            for error in errors:
                error_msg += f"\n     Error: {error}"
            validation_errors.append(error_msg)

    # Create report
    report_lines = [
        f"\n{'='*80}",
        f"Webhook Configuration Validation Report for {provider.upper()}",
        f"{'='*80}",
        f"Files checked: {len(webhook_files[provider])}",
        f"Webhook blocks found: {len(block_locations)}",
        f"Configurations found: {len(all_configs)}",
    ]

    if block_locations:
        report_lines.append("\n📦 Webhook Blocks Found:")
        for block_name, (file, line) in sorted(block_locations.items()):
            report_lines.append(f"  - {block_name} ({file}:{line})")

    if all_configs:
        report_lines.append("\n🔧 Webhook Configurations:")
        for config in all_configs:
            report_lines.append(
                f"  - {config['type']} in {config['file']}:"
                f"\n    Provider: {config.get('provider', 'N/A')}"
                f"\n    Type: {config.get('webhook_type', 'N/A')}"
                f"\n    Resource: {config.get('resource_format', 'N/A')}"
            )

    if validation_errors:
        report_lines.append(f"\n❌ VALIDATION ERRORS ({len(validation_errors)}):")
        report_lines.extend(validation_errors)
    else:
        report_lines.append("\n✅ All webhook configurations are valid!")

    if not spec:
        report_lines.append(
            f"\n⚠️  WARNING: No webhook specification found for {provider}. "
            f"Consider adding webhook configuration to test_data/{provider}.json"
        )

    # Summary
    report_lines.extend(
        [
            f"\n{'='*80}",
            f"Summary: {len(all_configs) - len(validation_errors)}/{len(all_configs)} configurations valid",
            f"{'='*80}\n",
        ]
    )

    # Print report
    report = "\n".join(report_lines)
    print(report)

    # Fail if there are validation errors
    if validation_errors:
        pytest.fail(
            f"Found {len(validation_errors)} invalid webhook configurations. See report above."
        )


def test_webhook_event_types():
    """Test that webhook blocks properly define their event types."""
    webhook_files = get_all_webhook_files()

    issues = []

    for provider, files in webhook_files.items():
        for py_file in files:
            with open(py_file, "r") as f:
                content = f.read()

            # Check for EventsFilter classes
            event_filter_pattern = (
                r"class\s+EventsFilter\s*\(.*\):([\s\S]*?)(?=class|\Z)"
            )
            matches = re.finditer(event_filter_pattern, content)

            for match in matches:
                class_content = match.group(1)

                # Extract event fields
                field_pattern = r"(\w+)\s*:\s*bool\s*="
                fields = re.findall(field_pattern, class_content)

                # Check that there are event fields defined
                if not fields:
                    issues.append(
                        f"{provider}/{py_file.name}: EventsFilter class has no event fields defined"
                    )

                # Check field naming conventions
                for field in fields:
                    if not field.islower() or not field.replace("_", "").isalnum():
                        issues.append(
                            f"{provider}/{py_file.name}: Event field '{field}' "
                            "doesn't follow naming convention (lowercase with underscores)"
                        )

    if issues:
        report = "\n".join(
            ["\nWebhook Event Type Issues:"] + [f"  - {issue}" for issue in issues]
        )
        pytest.fail(report)


def test_webhook_blocks_have_proper_structure():
    """Test that webhook blocks follow the expected structure."""
    webhook_files = get_all_webhook_files()

    structural_issues = []

    for provider, files in webhook_files.items():
        for py_file in files:
            with open(py_file, "r") as f:
                content = f.read()

            lines = content.split("\n")
            blocks = extract_webhook_blocks(content)

            for block_name, line_num in blocks:
                # For structural checks, look at the entire file content after the class definition
                # This is more reliable than trying to extract just the class content
                class_line_idx = line_num - 1
                remaining_content = "\n".join(lines[class_line_idx:])

                # Check for required components
                checks = [
                    ("BlockType.WEBHOOK", "block_type set to WEBHOOK", False),
                    ("class Input", "Input schema defined", True),
                    ("class Output", "Output schema defined", True),
                    (
                        "payload.*InputField|payload.*SchemaField",
                        "payload field in Input",
                        True,
                    ),
                    (
                        "webhook_url.*InputField|webhook_url.*SchemaField",
                        "webhook_url field in Input",
                        False,
                    ),
                    ("async def run", "async run method defined", True),
                ]

                for pattern, description, required in checks:
                    if required and not re.search(pattern, remaining_content):
                        structural_issues.append(
                            f"{provider}/{py_file.name}:{line_num} - "
                            f"{block_name} missing {description}"
                        )

    if structural_issues:
        report = "\n".join(
            ["\nWebhook Block Structure Issues:"]
            + [f"  - {issue}" for issue in structural_issues]
        )
        pytest.fail(report)


def test_webhook_specs_completeness():
    """Test that webhook specifications in JSON files are complete."""
    test_data_dir = Path(__file__).parent / "test_data"

    issues = []

    for spec_file in test_data_dir.glob("*.json"):
        with open(spec_file, "r") as f:
            spec = json.load(f)

        provider = spec_file.stem

        # Check if provider has webhook blocks
        webhook_files = get_all_webhook_files()
        if provider in webhook_files:
            # Provider has webhook blocks, check if spec has webhook section
            if "webhooks" not in spec:
                issues.append(
                    f"{provider}.json: Missing 'webhooks' section but provider has webhook blocks"
                )
            else:
                webhook_spec = spec["webhooks"]

                # Check webhook spec completeness
                recommended_fields = [
                    "allowed_webhook_types",
                    "resource_format_pattern",
                    "event_types",
                    "description",
                ]
                missing = [f for f in recommended_fields if f not in webhook_spec]

                if missing:
                    issues.append(
                        f"{provider}.json: Webhook spec missing recommended fields: "
                        f"{', '.join(missing)}"
                    )

    if issues:
        report = "\n".join(
            ["\nWebhook Specification Issues:"] + [f"  - {issue}" for issue in issues]
        )
        print(report)  # Just warn, don't fail
