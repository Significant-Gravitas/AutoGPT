#!/usr/bin/env python3
"""
Add cache configuration to a resolved docker-compose file for all services
that have a build key.
"""

import argparse

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Add cache config to a resolved compose file"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source compose file to read (should be output of `docker compose config`)",
    )
    parser.add_argument(
        "--cache-from",
        default="type=gha",
        help="Cache source configuration",
    )
    parser.add_argument(
        "--cache-to",
        default="type=gha,mode=max",
        help="Cache destination configuration",
    )
    parser.add_argument(
        "--backend-scope",
        default="",
        help="GHA cache scope for backend services (e.g., platform-backend-{hash})",
    )
    parser.add_argument(
        "--frontend-scope",
        default="",
        help="GHA cache scope for frontend service (e.g., platform-frontend-{hash})",
    )
    args = parser.parse_args()

    with open(args.source, "r") as f:
        compose = yaml.safe_load(f)

    modified_services = []
    for service_name, service_config in compose.get("services", {}).items():
        if "build" not in service_config:
            continue

        cache_from = args.cache_from
        cache_to = args.cache_to

        # Determine scope based on Dockerfile path
        if "type=gha" in args.cache_from or "type=gha" in args.cache_to:
            dockerfile = service_config["build"].get("dockerfile", "Dockerfile")
            if "frontend" in dockerfile:
                scope = args.frontend_scope
            elif "backend" in dockerfile:
                scope = args.backend_scope
            else:
                # Skip services that don't clearly match frontend/backend
                continue

            if scope:
                if "type=gha" in args.cache_from:
                    cache_from = f"{args.cache_from},scope={scope}"
                if "type=gha" in args.cache_to:
                    cache_to = f"{args.cache_to},scope={scope}"

        service_config["build"]["cache_from"] = [cache_from]
        service_config["build"]["cache_to"] = [cache_to]
        modified_services.append(service_name)

    # Write back to the same file
    with open(args.source, "w") as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

    print(f"Added cache config to {len(modified_services)} services in {args.source}:")
    for svc in modified_services:
        print(f"  - {svc}")


if __name__ == "__main__":
    main()
