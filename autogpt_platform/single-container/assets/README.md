# Docker Hub artwork

`dockerhub-banner.svg` is the editable source for `dockerhub-banner.png`. It
reuses the AutoGPT logo and the current Frankie, Maria, and Max portraits from
`frontend/public`. The PNG lets Docker Hub display the complete artwork without
loading the SVG's relative image references.

From the repository root, with DejaVu Sans installed, regenerate it with:

```sh
uvx --from cairosvg==2.8.2 cairosvg --unsafe \
  autogpt_platform/single-container/assets/dockerhub-banner.svg \
  -o autogpt_platform/single-container/assets/dockerhub-banner.png
```

`--unsafe` permits CairoSVG to load the trusted, repository-local images. Use it
only with this source. Keep the output below the repository's 500 KiB limit.

The parent README uses a raw GitHub URL pinned to the artwork's commit. When
updating the artwork, commit it first, then update that URL to the new commit so
the Docker Hub overview and GitHub preview show the same version.
