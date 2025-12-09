"""
URL and domain validation utilities.

Common URL validation operations used across the codebase.
"""


def matches_domain_pattern(hostname: str, domain_pattern: str) -> bool:
    """
    Check if a hostname matches a domain pattern.

    Supports wildcard patterns (*.example.com) which match:
    - The base domain (example.com)
    - Any subdomain (sub.example.com, deep.sub.example.com)

    Args:
        hostname: The hostname to check (e.g., "api.example.com")
        domain_pattern: The pattern to match against (e.g., "*.example.com" or "example.com")

    Returns:
        True if the hostname matches the pattern
    """
    hostname = hostname.lower()
    domain_pattern = domain_pattern.lower()

    if domain_pattern.startswith("*."):
        # Wildcard domain - matches base and any subdomains
        base_domain = domain_pattern[2:]
        return hostname == base_domain or hostname.endswith("." + base_domain)

    # Exact match
    return hostname == domain_pattern


def hostname_matches_any_domain(hostname: str, allowed_domains: list[str]) -> bool:
    """
    Check if a hostname matches any of the allowed domain patterns.

    Args:
        hostname: The hostname to check
        allowed_domains: List of allowed domain patterns (supports wildcards)

    Returns:
        True if the hostname matches any pattern
    """
    return any(matches_domain_pattern(hostname, domain) for domain in allowed_domains)
