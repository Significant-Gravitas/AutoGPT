"""
Pytest-based API endpoint validation for all provider blocks.

This test automatically discovers all API endpoints in provider implementations
and validates them against the JSON specifications in test_data/.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest


def extract_api_endpoints(file_content: str, provider: str) -> Set[Tuple[str, int]]:
    """
    Extract API endpoints from file content based on provider patterns.
    Returns tuples of (endpoint, line_number) for better error reporting.
    """
    endpoints = set()
    lines = file_content.split("\n")

    # Pattern 1: Direct URL strings in Requests() calls
    url_patterns = [
        # await Requests().get("https://...")
        (r'Requests\(\)\.\w+\(\s*["\']([^"\']+)["\']', "direct_call"),
        # await Requests().get(f"https://...")
        (r'Requests\(\)\.\w+\(\s*f["\']([^"\']+)["\']', "f_string_call"),
        # response = await Requests().get
        (r'await\s+Requests\(\)\.\w+\(\s*["\']([^"\']+)["\']', "await_call"),
        (r'await\s+Requests\(\)\.\w+\(\s*f["\']([^"\']+)["\']', "await_f_string"),
        # Requests().request(method, url)
        (r'Requests\(\)\.request\([^,]+,\s*["\']([^"\']+)["\']', "request_method"),
        (r'Requests\(\)\.request\([^,]+,\s*f["\']([^"\']+)["\']', "request_f_string"),
    ]

    # Pattern 2: URL variable assignments (for Exa style)
    url_var_patterns = [
        (r'url\s*=\s*["\']([^"\']+)["\']', "url_assignment"),
        (r'url\s*=\s*f["\']([^"\']+)["\']', "url_f_string"),
    ]

    # Check all patterns line by line for better error reporting
    for line_num, line in enumerate(lines, 1):
        # Check URL patterns
        for pattern, _ in url_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if match.startswith("http"):
                    endpoints.add((match, line_num))

        # Check URL variable patterns
        for pattern, _ in url_var_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if match.startswith("http"):
                    endpoints.add((match, line_num))

    # Pattern 3: Special handling for providers
    if provider == "gem":
        # Match endpoint parameters in make_request calls
        for line_num, line in enumerate(lines, 1):
            endpoint_match = re.search(r'endpoint\s*=\s*["\']([^"\']+)["\']', line)
            if endpoint_match:
                endpoint = endpoint_match.group(1)
                if endpoint.startswith("/"):
                    endpoints.add((f"https://api.gem.com{endpoint}", line_num))

    elif provider == "oxylabs":
        # Look for Oxylabs-specific URLs
        oxylabs_patterns = [
            (
                r'url\s*=\s*["\']https://realtime\.oxylabs\.io/v1/queries["\']',
                "realtime",
            ),
            (r'url\s*=\s*["\']https://data\.oxylabs\.io/v1/queries["\']', "data"),
            (r'url="https://data\.oxylabs\.io/v1/queries/batch"', "batch"),
            (r'f"https://data\.oxylabs\.io/v1/queries/{[^}]+}"', "job_status"),
            (r'f"https://data\.oxylabs\.io/v1/queries/{[^}]+}/results"', "job_results"),
            (r'"https://data\.oxylabs\.io/v1/info/callbacker_ips"', "callbacker"),
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern, endpoint_type in oxylabs_patterns:
                if re.search(pattern, line):
                    # Extract and normalize the URL
                    if endpoint_type == "realtime":
                        endpoints.add(
                            ("https://realtime.oxylabs.io/v1/queries", line_num)
                        )
                    elif endpoint_type == "data":
                        endpoints.add(("https://data.oxylabs.io/v1/queries", line_num))
                    elif endpoint_type == "batch":
                        endpoints.add(
                            ("https://data.oxylabs.io/v1/queries/batch", line_num)
                        )
                    elif endpoint_type == "job_status":
                        endpoints.add(
                            ("https://data.oxylabs.io/v1/queries/{job_id}", line_num)
                        )
                    elif endpoint_type == "job_results":
                        endpoints.add(
                            (
                                "https://data.oxylabs.io/v1/queries/{job_id}/results",
                                line_num,
                            )
                        )
                    elif endpoint_type == "callbacker":
                        endpoints.add(
                            ("https://data.oxylabs.io/v1/info/callbacker_ips", line_num)
                        )

    # Filter out invalid endpoints
    filtered_endpoints = set()
    for endpoint, line_num in endpoints:
        # Skip template placeholders and bare domains
        if "{base_url}" in endpoint or endpoint.endswith(
            (".com", ".io", ".com/", ".io/")
        ):
            continue
        # Skip non-URLs
        if not endpoint.startswith("http"):
            continue
        filtered_endpoints.add((endpoint, line_num))

    return filtered_endpoints


def normalize_endpoint_for_matching(endpoint: str) -> str:
    """Normalize endpoint for pattern matching."""
    # Replace specific IDs with placeholders
    endpoint = re.sub(r"/[a-f0-9-]{36}", "/{id}", endpoint)  # UUIDs
    endpoint = re.sub(r"/\d+", "/{id}", endpoint)  # Numeric IDs
    endpoint = re.sub(r"/[A-Z0-9_]+", "/{id}", endpoint)  # Uppercase IDs
    return endpoint


def match_endpoint_to_spec(
    endpoint: str, spec_endpoints: List[Dict]
) -> Tuple[bool, str]:
    """
    Check if an endpoint matches any pattern in the spec.
    Returns (is_match, matched_pattern or error_message)
    """
    for spec_endpoint in spec_endpoints:
        pattern = spec_endpoint["url_pattern"]

        # Direct match
        if endpoint == pattern:
            return True, pattern

        # Pattern matching with placeholders
        # Convert {param} to regex
        regex_pattern = pattern
        for placeholder in re.findall(r"\{([^}]+)\}", pattern):
            regex_pattern = regex_pattern.replace(f"{{{placeholder}}}", r"[^/]+")
        regex_pattern = f"^{regex_pattern}$"

        if re.match(regex_pattern, endpoint):
            return True, pattern

        # Try normalized matching
        normalized = normalize_endpoint_for_matching(endpoint)
        if re.match(regex_pattern, normalized):
            return True, pattern

    return False, f"No matching pattern found for: {endpoint}"


def get_all_provider_files() -> Dict[str, List[Path]]:
    """Get all Python files for each provider."""
    # Navigate from test/blocks to backend/blocks
    test_dir = Path(__file__).parent
    backend_dir = test_dir.parent.parent
    blocks_dir = backend_dir / "backend" / "blocks"
    providers = ["airtable", "baas", "elevenlabs", "exa", "gem", "oxylabs"]

    provider_files = {}
    for provider in providers:
        provider_dir = blocks_dir / provider
        if provider_dir.exists():
            files = [
                f
                for f in provider_dir.glob("*.py")
                if not f.name.startswith("_") and f.name != "__init__.py"
            ]
            provider_files[provider] = files

    return provider_files


def load_provider_spec(provider: str) -> Dict:
    """Load provider specification from JSON file."""
    # test_data is now in the same directory as this file
    spec_file = Path(__file__).parent / "test_data" / f"{provider}.json"

    if not spec_file.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_file}")

    with open(spec_file, "r") as f:
        return json.load(f)


@pytest.mark.parametrize(
    "provider", ["airtable", "baas", "elevenlabs", "exa", "gem", "oxylabs"]
)
def test_provider_api_endpoints(provider: str):
    """
    Test that all API endpoints in provider implementations match the specification.

    This test:
    1. Discovers all API endpoints in the provider's code
    2. Loads the expected endpoints from the JSON specification
    3. Validates that every endpoint in code has a matching pattern in the spec
    4. Reports any endpoints that don't match or are missing from the spec
    """
    # Get all files for this provider
    provider_files = get_all_provider_files()
    if provider not in provider_files:
        pytest.skip(f"Provider directory not found: {provider}")

    # Load the specification
    try:
        spec = load_provider_spec(provider)
    except FileNotFoundError as e:
        pytest.fail(str(e))

    # Extract all endpoints from code
    all_endpoints = set()
    endpoint_locations = {}  # endpoint -> [(file, line_num), ...]

    for py_file in provider_files[provider]:
        with open(py_file, "r") as f:
            content = f.read()
            endpoints = extract_api_endpoints(content, provider)

            for endpoint, line_num in endpoints:
                all_endpoints.add(endpoint)
                if endpoint not in endpoint_locations:
                    endpoint_locations[endpoint] = []
                endpoint_locations[endpoint].append((py_file.name, line_num))

    # Get expected endpoints from spec
    spec_endpoints = spec.get("api_calls", [])
    spec_patterns = [e["url_pattern"] for e in spec_endpoints]

    # Validate all discovered endpoints
    validation_errors = []
    unmatched_endpoints = []

    for endpoint in sorted(all_endpoints):
        is_match, result = match_endpoint_to_spec(endpoint, spec_endpoints)

        if not is_match:
            locations = endpoint_locations[endpoint]
            location_str = ", ".join([f"{file}:{line}" for file, line in locations])
            validation_errors.append(
                f"\n  ❌ Endpoint not in spec: {endpoint}\n"
                f"     Found at: {location_str}\n"
                f"     Reason: {result}"
            )
            unmatched_endpoints.append(endpoint)

    # Check for unused spec endpoints (warnings, not errors)
    unused_patterns = []
    for pattern in spec_patterns:
        pattern_used = False
        for endpoint in all_endpoints:
            is_match, _ = match_endpoint_to_spec(endpoint, [{"url_pattern": pattern}])
            if is_match:
                pattern_used = True
                break

        if not pattern_used:
            unused_patterns.append(pattern)

    # Create detailed report
    report_lines = [
        f"\n{'='*80}",
        f"API Endpoint Validation Report for {provider.upper()}",
        f"{'='*80}",
        f"Files checked: {len(provider_files[provider])}",
        f"Total endpoints found: {len(all_endpoints)}",
        f"Spec patterns: {len(spec_patterns)}",
    ]

    if validation_errors:
        report_lines.append(
            f"\n❌ VALIDATION ERRORS ({len(validation_errors)} endpoints don't match spec):"
        )
        report_lines.extend(validation_errors)
    else:
        report_lines.append("\n✅ All endpoints match specification!")

    if unused_patterns:
        report_lines.append(f"\n⚠️  UNUSED SPEC PATTERNS ({len(unused_patterns)}):")
        for pattern in unused_patterns:
            report_lines.append(f"  - {pattern}")
        report_lines.append(
            "  These patterns are defined in the spec but not found in code."
        )

    # Summary
    report_lines.extend(
        [
            f"\n{'='*80}",
            f"Summary: {len(all_endpoints) - len(unmatched_endpoints)}/{len(all_endpoints)} endpoints valid",
            f"{'='*80}\n",
        ]
    )

    # Print the full report
    report = "\n".join(report_lines)
    print(report)

    # Fail if there are validation errors
    if validation_errors:
        pytest.fail(
            f"Found {len(validation_errors)} endpoints that don't match the specification. See report above."
        )


def test_all_providers_have_specs():
    """Test that all provider directories have corresponding JSON specifications."""
    # Navigate from test/blocks to backend/blocks
    test_dir = Path(__file__).parent
    backend_dir = test_dir.parent.parent
    blocks_dir = backend_dir / "backend" / "blocks"
    # test_data is now in the test directory
    test_data_dir = test_dir / "test_data"

    # Find all provider directories
    provider_dirs = [
        d.name
        for d in blocks_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(("_", "."))
        and d.name != "test_data"
        and (d / "blocks.py").exists()  # Only directories with blocks.py
    ]

    # Check each has a spec
    missing_specs = []
    for provider in provider_dirs:
        spec_file = test_data_dir / f"{provider}.json"
        if not spec_file.exists():
            missing_specs.append(provider)

    if missing_specs:
        pytest.fail(
            f"Missing JSON specifications for providers: {', '.join(missing_specs)}"
        )


def test_spec_json_validity():
    """Test that all JSON specification files are valid and have required fields."""
    # test_data is now in the test directory
    test_data_dir = Path(__file__).parent / "test_data"

    spec_files = list(test_data_dir.glob("*.json"))

    for spec_file in spec_files:
        # Load and validate JSON
        try:
            with open(spec_file, "r") as f:
                spec = json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {spec_file.name}: {e}")

        # Check required fields
        required_fields = ["provider", "auth_type", "api_calls"]
        missing_fields = [f for f in required_fields if f not in spec]

        if missing_fields:
            pytest.fail(
                f"{spec_file.name} missing required fields: {', '.join(missing_fields)}"
            )

        # Validate api_calls structure
        for i, call in enumerate(spec.get("api_calls", [])):
            required_call_fields = ["name", "method", "url_pattern"]
            missing = [f for f in required_call_fields if f not in call]

            if missing:
                pytest.fail(
                    f"{spec_file.name}: api_calls[{i}] missing required fields: {', '.join(missing)}"
                )
