#!/usr/bin/env python3
"""
Add cache configuration to a resolved docker-compose file for all services
that have a build key, and ensure image names match what docker compose expects.
"""

import argparse

import yaml


DEFAULT_BRANCH = "dev"
CACHE_BUILDS_FOR_COMPONENTS = ["backend", "frontend"]


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
    for component in CACHE_BUILDS_FOR_COMPONENTS:
        parser.add_argument(
            f"--{component}-hash",
            default="",
            help=f"Hash for {component} cache scope (e.g., from hashFiles())",
        )
    parser.add_argument(
        "--git-ref",
        default="",
        help="Git ref for branch-based cache scope (e.g., refs/heads/master)",
    )
    args = parser.parse_args()

    # Normalize git ref to a safe scope name (e.g., refs/heads/master -> master)
    git_ref_scope = ""
    if args.git_ref:
        git_ref_scope = args.git_ref.replace("refs/heads/", "").replace("/", "-")

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

        # This service will do the actual build - add cache config
        cache_from_list = []
        cache_to_list = []

        component = get_component(dockerfile)
        if not component:
            # Skip services that don't clearly match frontend/backend
            continue

        # Get the hash for this component
        component_hash = getattr(args, f"{component}_hash")

        # Scope format: platform-{component}-{target}-{hash|ref}
        # Example: platform-backend-server-abc123

        if "type=gha" in args.cache_from:
            # 1. Primary: exact hash match (most specific)
            if component_hash:
                hash_scope = f"platform-{component}-{target}-{component_hash}"
                cache_from_list.append(f"{args.cache_from},scope={hash_scope}")

            # 2. Fallback: branch-based cache
            if git_ref_scope:
                ref_scope = f"platform-{component}-{target}-{git_ref_scope}"
                cache_from_list.append(f"{args.cache_from},scope={ref_scope}")

            # 3. Fallback: dev branch cache (for PRs/feature branches)
            if git_ref_scope and git_ref_scope != DEFAULT_BRANCH:
                master_scope = f"platform-{component}-{target}-{DEFAULT_BRANCH}"
                cache_from_list.append(f"{args.cache_from},scope={master_scope}")

        if "type=gha" in args.cache_to:
            # Write to both hash-based and branch-based scopes
            if component_hash:
                hash_scope = f"platform-{component}-{target}-{component_hash}"
                cache_to_list.append(f"{args.cache_to},scope={hash_scope}")

            if git_ref_scope:
                ref_scope = f"platform-{component}-{target}-{git_ref_scope}"
                cache_to_list.append(f"{args.cache_to},scope={ref_scope}")

        # Ensure we have at least one cache source/target
        if not cache_from_list:
            cache_from_list.append(args.cache_from)
        if not cache_to_list:
            cache_to_list.append(args.cache_to)

        build_config["cache_from"] = cache_from_list
        build_config["cache_to"] = cache_to_list
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
