#!/usr/bin/env python3

import hashlib
import os


def cache_settings(
    event: str, ref: str, image: str, arch: str, *, head_ref: str = ""
) -> list[str]:
    if arch not in {"amd64", "arm64"}:
        raise ValueError(f"Unsupported cache architecture: {arch}")
    shared = f"{image}:v4-{arch}"
    imports = [shared]
    export = None
    if event == "workflow_dispatch" or (event == "pull_request" and head_ref):
        branch_ref = f"refs/heads/{head_ref}" if event == "pull_request" else ref
        branch_digest = hashlib.sha256(branch_ref.encode()).hexdigest()[:20]
        branch_cache = f"{image}:v4-branch-{branch_digest}-{arch}"
        imports.insert(0, branch_cache)
        if event == "workflow_dispatch":
            export = branch_cache
    elif event == "push" and ref == "refs/heads/dev":
        export = shared
    settings = [
        f"{target}.cache-from=type=registry,ref={cache_ref}"
        for target in ("backend-server-base", "backend-server", "single-container")
        for cache_ref in imports
    ]
    if export:
        settings.append(
            f"single-container.cache-to=type=registry,ref={export},mode=max,"
            "oci-mediatypes=true,image-manifest=false,ignore-error=true"
        )
    return settings


if __name__ == "__main__":
    settings = cache_settings(
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REF"],
        os.environ["BUILD_CACHE_IMAGE"],
        os.environ["BUILD_CACHE_ARCH"],
        head_ref=os.environ.get("GITHUB_HEAD_REF", ""),
    )
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write("set<<CACHE_EOF\n" + "\n".join(settings) + "\nCACHE_EOF\n")
