#!/usr/bin/env python3
"""
Generate a docker-compose.ci.yml with cache configuration for all services
that have a build key in the source compose file.
"""

import argparse

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Generate docker-compose cache override file"
    )
    parser.add_argument(
        "--source",
        default="docker-compose.platform.yml",
        help="Source compose file to read (default: docker-compose.platform.yml)",
    )
    parser.add_argument(
        "--output",
        default="docker-compose.ci.yml",
        help="Output compose file to write (default: docker-compose.ci.yml)",
    )
    parser.add_argument(
        "--cache-from",
        default="type=local,src=/tmp/.buildx-cache",
        help="Cache source configuration",
    )
    parser.add_argument(
        "--cache-to",
        default="type=local,dest=/tmp/.buildx-cache-new,mode=max",
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

    ci_compose = {"services": {}}
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

        ci_compose["services"][service_name] = {
            "build": {
                "cache_from": [cache_from],
                "cache_to": [cache_to],
            }
        }

    with open(args.output, "w") as f:
        yaml.dump(ci_compose, f, default_flow_style=False)

    services = list(ci_compose["services"].keys())
    print(f"Generated {args.output} with cache config for {len(services)} services:")
    for svc in services:
        print(f"  - {svc}")


if __name__ == "__main__":
    main()
