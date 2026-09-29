#!/usr/bin/env python3
"""
Add registry cache configuration to a resolved docker-compose file for all
services that have a build key, and ensure image names match what docker
compose expects.

Each build target gets its own tag in one registry cache image, e.g.
ghcr.io/org/buildcache:backend-server. Every run reads it; only runs given
--write-cache write it. Cache export errors are ignored, so a missing package
or permission costs a cold build, never a failed one.
"""

import argparse

import yaml


CACHE_BUILDS_FOR_COMPONENTS = ["backend", "frontend"]
CACHE_TO_OPTIONS = "mode=max,oci-mediatypes=true,image-manifest=false,ignore-error=true"


def main():
    parser = argparse.ArgumentParser(
        description="Add registry cache config to a resolved compose file"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source compose file to read (should be output of `docker compose config`)",
    )
    parser.add_argument(
        "--cache-image",
        required=True,
        help="Registry image that holds the build cache, one tag per build target",
    )
    parser.add_argument(
        "--write-cache",
        action="store_true",
        help="Also export the build cache to the registry (mode=max)",
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

    def get_component(dockerfile: str) -> str | None:
        """Get component name (frontend/backend) from dockerfile path."""
        for component in CACHE_BUILDS_FOR_COMPONENTS:
            if component in dockerfile:
                return component
        return None

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
        target = build_config.get("target", "latest")
        image_name = get_image_name(dockerfile, target)

        # Set image name for all services (needed for both builders and deduped)
        service_config["image"] = image_name

        if service_name in services_to_dedupe:
            # Remove build config - this service will use the pre-built image
            del service_config["build"]
            continue

        component = get_component(dockerfile)
        if not component:
            # Skip services that don't clearly match frontend/backend
            continue

        # Example: ghcr.io/significant-gravitas/autogpt-platform-e2e-buildcache:backend-server
        cache_ref = f"type=registry,ref={args.cache_image}:{component}-{target}"
        build_config["cache_from"] = [cache_ref]
        if args.write_cache:
            build_config["cache_to"] = [f"{cache_ref},{CACHE_TO_OPTIONS}"]
        else:
            build_config.pop("cache_to", None)
        modified_services.append(service_name)

    # Write back to the same file
    with open(args.source, "w") as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

    print(f"Added cache config to {len(modified_services)} services in {args.source}:")
    for svc in modified_services:
        svc_config = compose["services"][svc]
        build_cfg = svc_config.get("build", {})
        cache_from_list = build_cfg.get("cache_from", ["none"])
        cache_to_list = build_cfg.get("cache_to", ["none"])
        print(f"  - {svc}")
        print(f"      image: {svc_config.get('image', 'N/A')}")
        print(f"      cache_from: {cache_from_list}")
        print(f"      cache_to: {cache_to_list}")
    if services_to_dedupe:
        print(
            f"Deduplicated {len(services_to_dedupe)} services (will use pre-built images):"
        )
        for svc in services_to_dedupe:
            print(f"  - {svc} -> {compose['services'][svc].get('image', 'N/A')}")


if __name__ == "__main__":
    main()
