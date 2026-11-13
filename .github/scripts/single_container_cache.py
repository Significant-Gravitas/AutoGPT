#!/usr/bin/env python3

import hashlib
import os


def cache_settings(event: str, ref: str, image: str, arch: str) -> list[str]:
    if arch not in {"amd64", "arm64"}:
        raise ValueError(f"Unsupported cache architecture: {arch}")
    shared = f"{image}:v4-{arch}"
    imports = [shared]
    export = None
    if event == "workflow_dispatch":
        branch_digest = hashlib.sha256(ref.encode()).hexdigest()[:20]
        export = f"{image}:v4-branch-{branch_digest}-{arch}"
        imports.insert(0, export)
    elif event == "push" and ref == "refs/heads/dev":
        export = shared
    # Only single-container reads the cache. Its build already contains the
    # backend-server-base and backend-server stages, so their layers still come
    # from the cache. If those helper targets read it too, the first build to
    # load a shared stage owns its cache import, and the layers it later pulls
    # authenticate through that build's session. The helper builds finish
    # within a minute, and once their registry token expires the pulls fail
    # with "no active session".
    settings = [
        f"single-container.cache-from=type=registry,ref={cache_ref}"
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
    )
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write("set<<CACHE_EOF\n" + "\n".join(settings) + "\nCACHE_EOF\n")
