#!/usr/bin/env python3

import hashlib
import os


def cache_settings(event: str, ref: str, image: str, arch: str) -> list[str]:
    if arch not in {"amd64", "arm64"}:
        raise ValueError(f"Unsupported cache architecture: {arch}")
    if event == "pull_request":
        return []
    trusted = f"{image}:v4-{arch}"
    imports = [trusted]
    export = None
    if event == "workflow_dispatch":
        branch_digest = hashlib.sha256(ref.encode()).hexdigest()[:20]
        export = f"{image}:v4-branch-{branch_digest}-{arch}"
        imports.insert(0, export)
    elif event == "push" and ref == "refs/heads/dev":
        export = trusted
    settings = [
        f"{target}.cache-from=type=registry,ref={cache_ref}"
        for target in ("backend-server", "single-container")
        for cache_ref in imports
    ]
    if export:
        settings.append(
            f"single-container.cache-to=type=registry,ref={export},mode=max,"
            "oci-mediatypes=true,image-manifest=false"
        )
    return settings


if __name__ == "__main__":
    settings = cache_settings(
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REF"],
        os.environ["BUILD_CACHE_IMAGE"],
        os.environ["BUILD_CACHE_ARCH"],
    )
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write("set<<CACHE_EOF\n" + "\n".join(settings) + "\nCACHE_EOF\n")
