"""
Test script for Linear GraphQL API - Customer Requests operations.

Tests the exact GraphQL calls needed for:
1. search_feature_requests - search issues in the Customer Feature Requests project
2. add_feature_request - upsert customer + create customer need on issue

Requires LINEAR_API_KEY in backend/.env
Generate one at: https://linear.app/settings/api
"""

import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

LINEAR_API_URL = "https://api.linear.app/graphql"
API_KEY = os.getenv("LINEAR_API_KEY")

# Target project for feature requests
FEATURE_REQUEST_PROJECT_ID = "13f066f3-f639-4a67-aaa3-31483ebdf8cd"
# Team: Internal
TEAM_ID = "557fd3d5-087e-43a9-83e3-476c8313ce49"

if not API_KEY:
    print("ERROR: LINEAR_API_KEY not found in .env")
    print("Generate a personal API key at: https://linear.app/settings/api")
    print("Then add LINEAR_API_KEY=lin_api_... to backend/.env")
    sys.exit(1)

HEADERS = {
    "Authorization": API_KEY,
    "Content-Type": "application/json",
}


def graphql(query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query against Linear API."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    resp = httpx.post(LINEAR_API_URL, json=payload, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    data = resp.json()

    if "errors" in data:
        print(f"GraphQL Errors: {json.dumps(data['errors'], indent=2)}")

    return data


# ---------------------------------------------------------------------------
# QUERIES
# ---------------------------------------------------------------------------

# Search issues within the feature requests project by title/description
SEARCH_ISSUES_IN_PROJECT = """
query SearchFeatureRequests($filter: IssueFilter!, $first: Int) {
  issues(filter: $filter, first: $first) {
    nodes {
      id
      identifier
      title
      description
      url
      state {
        name
        type
      }
      project {
        id
        name
      }
      labels {
        nodes {
          name
        }
      }
    }
  }
}
"""

# Get issue with its customer needs
GET_ISSUE_WITH_NEEDS = """
query GetIssueWithNeeds($id: String!) {
  issue(id: $id) {
    id
    identifier
    title
    url
    needs {
      nodes {
        id
        body
        priority
        customer {
          id
          name
          domains
          externalIds
        }
      }
    }
  }
}
"""

# Search customers
SEARCH_CUSTOMERS = """
query SearchCustomers($filter: CustomerFilter, $first: Int) {
  customers(filter: $filter, first: $first) {
    nodes {
      id
      name
      domains
      externalIds
      revenue
      size
      status {
        name
      }
      tier {
        name
      }
    }
  }
}
"""

# ---------------------------------------------------------------------------
# MUTATIONS
# ---------------------------------------------------------------------------

CUSTOMER_UPSERT = """
mutation CustomerUpsert($input: CustomerUpsertInput!) {
  customerUpsert(input: $input) {
    success
    customer {
      id
      name
      domains
      externalIds
    }
  }
}
"""

CUSTOMER_NEED_CREATE = """
mutation CustomerNeedCreate($input: CustomerNeedCreateInput!) {
  customerNeedCreate(input: $input) {
    success
    need {
      id
      body
      priority
      customer {
        id
        name
      }
      issue {
        id
        identifier
        title
      }
    }
  }
}
"""

ISSUE_CREATE = """
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue {
      id
      identifier
      title
      url
    }
  }
}
"""


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------


def test_1_search_feature_requests():
    """Search for feature requests in the target project by keyword."""
    print("\n" + "=" * 60)
    print("TEST 1: Search feature requests in project by keyword")
    print("=" * 60)

    search_term = "agent"
    result = graphql(
        SEARCH_ISSUES_IN_PROJECT,
        {
            "filter": {
                "project": {"id": {"eq": FEATURE_REQUEST_PROJECT_ID}},
                "or": [
                    {"title": {"containsIgnoreCase": search_term}},
                    {"description": {"containsIgnoreCase": search_term}},
                ],
            },
            "first": 5,
        },
    )

    issues = result.get("data", {}).get("issues", {}).get("nodes", [])
    for issue in issues:
        proj = issue.get("project") or {}
        print(f"\n  [{issue['identifier']}] {issue['title']}")
        print(f"    Project: {proj.get('name', 'N/A')}")
        print(f"    State: {issue['state']['name']}")
        print(f"    URL: {issue['url']}")

    print(f"\n  Found {len(issues)} issues matching '{search_term}'")
    return issues


def test_2_list_all_in_project():
    """List all issues in the feature requests project."""
    print("\n" + "=" * 60)
    print("TEST 2: List all issues in Customer Feature Requests project")
    print("=" * 60)

    result = graphql(
        SEARCH_ISSUES_IN_PROJECT,
        {
            "filter": {
                "project": {"id": {"eq": FEATURE_REQUEST_PROJECT_ID}},
            },
            "first": 10,
        },
    )

    issues = result.get("data", {}).get("issues", {}).get("nodes", [])
    if not issues:
        print("  No issues in project yet (empty project)")
    for issue in issues:
        print(f"\n  [{issue['identifier']}] {issue['title']}")
        print(f"    State: {issue['state']['name']}")

    print(f"\n  Total: {len(issues)} issues")
    return issues


def test_3_search_customers():
    """List existing customers."""
    print("\n" + "=" * 60)
    print("TEST 3: List customers")
    print("=" * 60)

    result = graphql(SEARCH_CUSTOMERS, {"first": 10})
    customers = result.get("data", {}).get("customers", {}).get("nodes", [])

    if not customers:
        print("  No customers exist yet")
    for c in customers:
        status = c.get("status") or {}
        tier = c.get("tier") or {}
        print(f"\n  [{c['id'][:8]}...] {c['name']}")
        print(f"    Domains: {c.get('domains', [])}")
        print(f"    External IDs: {c.get('externalIds', [])}")
        print(
            f"    Status: {status.get('name', 'N/A')}, Tier: {tier.get('name', 'N/A')}"
        )

    print(f"\n  Total: {len(customers)} customers")
    return customers


def test_4_customer_upsert():
    """Upsert a test customer."""
    print("\n" + "=" * 60)
    print("TEST 4: Customer upsert (find-or-create)")
    print("=" * 60)

    result = graphql(
        CUSTOMER_UPSERT,
        {
            "input": {
                "name": "Test Customer (API Test)",
                "domains": ["test-api-customer.example.com"],
                "externalId": "test-customer-001",
            }
        },
    )

    upsert = result.get("data", {}).get("customerUpsert", {})
    if upsert.get("success"):
        customer = upsert["customer"]
        print(f"  Success! Customer: {customer['name']}")
        print(f"    ID: {customer['id']}")
        print(f"    Domains: {customer['domains']}")
        print(f"    External IDs: {customer['externalIds']}")
        return customer
    else:
        print(f"  Failed: {json.dumps(result, indent=2)}")
        return None


def test_5_create_issue_and_need(customer_id: str):
    """Create a new feature request issue and attach a customer need."""
    print("\n" + "=" * 60)
    print("TEST 5: Create issue + customer need")
    print("=" * 60)

    # Step 1: Create issue in the project
    result = graphql(
        ISSUE_CREATE,
        {
            "input": {
                "title": "Test Feature Request (API Test - safe to delete)",
                "description": "This is a test feature request created via the GraphQL API.",
                "teamId": TEAM_ID,
                "projectId": FEATURE_REQUEST_PROJECT_ID,
            }
        },
    )

    data = result.get("data")
    if not data:
        print(f"  Issue creation failed: {json.dumps(result, indent=2)}")
        return None
    issue_data = data.get("issueCreate", {})
    if not issue_data.get("success"):
        print(f"  Issue creation failed: {json.dumps(result, indent=2)}")
        return None

    issue = issue_data["issue"]
    print(f"  Created issue: [{issue['identifier']}] {issue['title']}")
    print(f"    URL: {issue['url']}")

    # Step 2: Attach customer need
    result = graphql(
        CUSTOMER_NEED_CREATE,
        {
            "input": {
                "customerId": customer_id,
                "issueId": issue["id"],
                "body": "Our team really needs this feature for our workflow. High priority for us!",
                "priority": 0,
            }
        },
    )

    need_data = result.get("data", {}).get("customerNeedCreate", {})
    if need_data.get("success"):
        need = need_data["need"]
        print(f"  Attached customer need: {need['id']}")
        print(f"    Customer: {need['customer']['name']}")
        print(f"    Body: {need['body'][:80]}")
    else:
        print(f"  Customer need creation failed: {json.dumps(result, indent=2)}")

    # Step 3: Verify by fetching the issue with needs
    print("\n  Verifying...")
    verify = graphql(GET_ISSUE_WITH_NEEDS, {"id": issue["id"]})
    issue_verify = verify.get("data", {}).get("issue", {})
    needs = issue_verify.get("needs", {}).get("nodes", [])
    print(f"  Issue now has {len(needs)} customer need(s)")
    for n in needs:
        cust = n.get("customer") or {}
        print(f"    - {cust.get('name', 'N/A')}: {n.get('body', '')[:60]}")

    return issue


def test_6_add_need_to_existing(customer_id: str, issue_id: str):
    """Add a customer need to an existing issue (the common case)."""
    print("\n" + "=" * 60)
    print("TEST 6: Add customer need to existing issue")
    print("=" * 60)

    result = graphql(
        CUSTOMER_NEED_CREATE,
        {
            "input": {
                "customerId": customer_id,
                "issueId": issue_id,
                "body": "We also want this! +1 from our organization.",
                "priority": 0,
            }
        },
    )

    need_data = result.get("data", {}).get("customerNeedCreate", {})
    if need_data.get("success"):
        need = need_data["need"]
        print(f"  Success! Need: {need['id']}")
        print(f"    Customer: {need['customer']['name']}")
        print(f"    Issue: [{need['issue']['identifier']}] {need['issue']['title']}")
        return need
    else:
        print(f"  Failed: {json.dumps(result, indent=2)}")
        return None


def main():
    print("Linear GraphQL API - Customer Requests Test Suite")
    print("=" * 60)
    print(f"API URL: {LINEAR_API_URL}")
    print(f"API Key: {API_KEY[:10]}...")
    print(f"Project: Customer Feature Requests ({FEATURE_REQUEST_PROJECT_ID[:8]}...)")

    # --- Read-only tests ---
    test_1_search_feature_requests()
    test_2_list_all_in_project()
    test_3_search_customers()

    # --- Write tests ---
    print("\n" + "=" * 60)
    answer = (
        input("Run WRITE tests? (creates test customer + issue + need) [y/N]: ")
        .strip()
        .lower()
    )
    if answer != "y":
        print("Skipped write tests.")
        print("\nDone!")
        return

    customer = test_4_customer_upsert()
    if not customer:
        print("Customer upsert failed, stopping.")
        return

    issue = test_5_create_issue_and_need(customer["id"])
    if not issue:
        print("Issue creation failed, stopping.")
        return

    # Test adding a second need to the same issue (simulates another customer requesting same feature)
    # First upsert a second customer
    result = graphql(
        CUSTOMER_UPSERT,
        {
            "input": {
                "name": "Second Test Customer",
                "domains": ["second-test.example.com"],
                "externalId": "test-customer-002",
            }
        },
    )
    customer2 = result.get("data", {}).get("customerUpsert", {}).get("customer")
    if customer2:
        test_6_add_need_to_existing(customer2["id"], issue["id"])

    print("\n" + "=" * 60)
    print("All tests complete!")
    print(
        "Check the project: https://linear.app/autogpt/project/customer-feature-requests-710dcbf8bf4e/issues"
    )


if __name__ == "__main__":
    main()
