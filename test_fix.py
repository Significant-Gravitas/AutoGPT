#!/usr/bin/env python3

import re


def prepare_pr_api_url(pr_url: str, path: str) -> str:
    pattern = r"^(?:https?://)?github\.com/([^/]+/[^/]+)/pull/(\d+)"
    match = re.match(pattern, pr_url)
    if not match:
        raise ValueError(f"Invalid GitHub PR URL: {pr_url}. URL must be a valid pull request URL, e.g., https://github.com/owner/repo/pull/123")

    repo_path, pr_number = match.groups()
    return f"{repo_path}/pulls/{pr_number}/{path}"


# Test cases
def test_prepare_pr_api_url():
    # Test valid PR URLs
    print("Testing valid PR URLs:")
    valid_urls = [
        "https://github.com/owner/repo/pull/123",
        "http://github.com/owner/repo/pull/456",
        "github.com/owner/repo/pull/789"
    ]
    
    for url in valid_urls:
        try:
            result = prepare_pr_api_url(url, "files")
            print(f"✓ {url} -> {result}")
        except Exception as e:
            print(f"✗ {url} -> ERROR: {e}")
    
    # Test invalid URLs (should raise ValueError)
    print("\nTesting invalid URLs (should raise ValueError):")
    invalid_urls = [
        "https://github.com/Significant-Gravitas/AutoGPT/issues/10313",
        "https://gitlab.com/owner/repo/pull/123",
        "https://github.com/owner/repo/issues/123",
        "https://github.com/owner/repo",
        "not a url at all"
    ]
    
    for url in invalid_urls:
        try:
            result = prepare_pr_api_url(url, "files")
            print(f"✗ {url} -> {result} (should have failed!)")
        except ValueError as e:
            print(f"✓ {url} -> Correctly raised ValueError: {e}")
        except Exception as e:
            print(f"✗ {url} -> Unexpected error: {e}")


if __name__ == "__main__":
    test_prepare_pr_api_url()