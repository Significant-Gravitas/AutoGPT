#!/usr/bin/env bash
# TEMPORARY (PR #14892 only, removed before review): measure the registry build cache
# on the free runner. PRs can't write GHCR, so a runner-local registry stands in for it:
# a cold bake that writes the cache, then fresh builders that only read it.
set -eo pipefail
cd "$GITHUB_WORKSPACE/autogpt_platform"
OUT="$RUNNER_TEMP/cache-measure"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.md"
CACHE_IMAGE=localhost:5000/e2e-buildcache

cat > "$OUT/buildkitd.toml" <<'EOF'
[registry."localhost:5000"]
  http = true
EOF

fresh_builder() {
  docker buildx rm measure >/dev/null 2>&1 || true
  docker buildx create --name measure --driver docker-container \
    --driver-opt "image=$BUILDKIT_IMAGE" --driver-opt network=host \
    --buildkitd-config "$OUT/buildkitd.toml" --bootstrap >/dev/null
}

prepare() {
  docker compose -f docker-compose.yml config > docker-compose.resolved.yml
  if ! grep -q "NEXT_PUBLIC_SOURCEMAPS" docker-compose.resolved.yml; then
    sed -i '/NEXT_PUBLIC_PW_TEST/a\        NEXT_PUBLIC_SOURCEMAPS: "true"' docker-compose.resolved.yml
  fi
  sed -i 's/^BETTER_AUTH_SECRET=.*/BETTER_AUTH_SECRET=/' frontend/.env
  python ../.github/workflows/scripts/docker-ci-fix-compose-build-cache.py \
    --source docker-compose.resolved.yml --cache-image "$CACHE_IMAGE" "$@"
}

bake_one() { # label, part, targets...
  local label=$1 part=$2 start
  shift 2
  start=$(date +%s)
  docker buildx --builder measure bake --progress=plain --allow=fs.read=.. \
    -f docker-compose.resolved.yml --load "$@" > "$OUT/$label-$part.log" 2>&1
  echo $(( $(date +%s) - start )) > "$OUT/$label-$part.secs"
}

# Both bakes run side by side, like the e2e job's background steps
bake() { # label
  local label=$1 backend
  backend=$(python -c "import yaml; s = yaml.safe_load(open('docker-compose.resolved.yml'))['services']; print(' '.join(n for n, c in s.items() if 'build' in c and n != 'frontend'))")
  bake_one "$label" backend $backend &
  local b=$!
  bake_one "$label" frontend --set frontend.args.NEXT_SKIP_BUILD_CHECKS=true frontend &
  local f=$!
  wait $b
  wait $f
  local exports cached
  exports=$(grep -hE "preparing build cache for export [0-9.]+s done" "$OUT/$label"-*.log | awk '{s += $(NF-1)} END {printf "%.0f", s}')
  cached=$(cat "$OUT/$label"-*.log | grep -c ' CACHED$' || true)
  echo "| $label | $(cat "$OUT/$label-backend.secs") s | $(cat "$OUT/$label-frontend.secs") s | ${exports:-0} s | $cached |" >> "$SUMMARY"
}

{
  echo "### Registry cache on ubuntu-latest (runner-local registry standing in for GHCR)"
  echo
  echo "| Run | Backend bake | Frontend bake | Cache export prep | CACHED steps |"
  echo "|---|---|---|---|---|"
} > "$SUMMARY"

fresh_builder
prepare --write-cache
bake cold-write

fresh_builder
prepare
bake warm-same-source

# A typical PR: one backend and one frontend source file changed
echo "# cache measurement" >> backend/backend/app.py
echo "// cache measurement" >> frontend/src/app/layout.tsx
fresh_builder
prepare
bake warm-both-changed

docker buildx rm measure >/dev/null 2>&1 || true
curl -s localhost:5000/v2/e2e-buildcache/tags/list >> "$SUMMARY"
cat "$SUMMARY" | tee -a "$GITHUB_STEP_SUMMARY"
