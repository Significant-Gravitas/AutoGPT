#!/usr/bin/env python3
"""
Add cache configuration to a resolved docker-compose file for all services
that have a build key, and ensure image names match what docker compose expects.
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

    # Get project name from compose file or default
    project_name = compose.get("name", "autogpt_platform")

    def get_image_name(dockerfile: str, target: str) -> str:
        """Generate image name based on Dockerfile folder and build target."""
        dockerfile_parts = dockerfile.replace("\\", "/").split("/")
        if len(dockerfile_parts) >= 2:
            folder_name = dockerfile_parts[-2]  # e.g., "backend" or "frontend"
        else:
            folder_name = "app"
        return f"{project_name}-{folder_name}:{target}"

    def get_build_key(dockerfile: str, target: str) -> str:
        """Generate a unique key for a Dockerfile+target combination."""
        return f"{dockerfile}:{target}"

    # First pass: collect all services with build configs and identify duplicates
    # Track which (dockerfile, target) combinations we've seen
    build_key_to_first_service: dict[str, str] = {}
    services_to_build: list[str] = []
    services_to_dedupe: list[str] = []

    for service_name, service_config in compose.get("services", {}).items():
        if "build" not in service_config:
            continue

        build_config = service_config["build"]
        dockerfile = build_config.get("dockerfile", "Dockerfile")
        target = build_config.get("target", "default")
        build_key = get_build_key(dockerfile, target)

        if build_key not in build_key_to_first_service:
            # First service with this build config - it will do the actual build
            build_key_to_first_service[build_key] = service_name
            services_to_build.append(service_name)
        else:
            # Duplicate - will just use the image from the first service
            services_to_dedupe.append(service_name)

    # Second pass: configure builds and deduplicate
    modified_services = []
    for service_name, service_config in compose.get("services", {}).items():
        if "build" not in service_config:
            continue

        build_config = service_config["build"]
        dockerfile = build_config.get("dockerfile", "Dockerfile")
        target = build_config.get("target", "default")
        image_name = get_image_name(dockerfile, target)

        # Set image name for all services (needed for both builders and deduped)
        service_config["image"] = image_name

        if service_name in services_to_dedupe:
            # Remove build config - this service will use the pre-built image
            del service_config["build"]
            continue

        # This service will do the actual build - add cache config
        cache_from = args.cache_from
        cache_to = args.cache_to

        # Determine scope based on Dockerfile path
        if "type=gha" in args.cache_from or "type=gha" in args.cache_to:
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

        build_config["cache_from"] = [cache_from]
        build_config["cache_to"] = [cache_to]
        modified_services.append(service_name)

    # Write back to the same file
    with open(args.source, "w") as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

    print(f"Added cache config to {len(modified_services)} services in {args.source}:")
    for svc in modified_services:
        print(f"  - {svc}")
    if services_to_dedupe:
        print(
            f"Deduplicated {len(services_to_dedupe)} services (will use pre-built images):"
        )
        for svc in services_to_dedupe:
            print(f"  - {svc}")


if __name__ == "__main__":
    main()
