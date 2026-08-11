#!/usr/bin/env bash

set -euo pipefail

platform_from_image_json() {
  local expected_arch="$1"
  jq -er --arg expected "linux/${expected_arch}" '
    if has("os") and has("architecture") then
      "\(.os)/\(.architecture)"
    else
      to_entries
      | map(select(.value.os? != null and .value.architecture? != null))
      | map("\(.value.os)/\(.value.architecture)")
      | if index($expected) != null then
          $expected
        elif length == 1 then
          .[0]
        else
          error("expected one matching runnable platform")
        end
    end
  '
}

repository_is_public() {
  jq -e '
    .namespace == "significantgravitas"
    and .name == "autogpt"
    and .is_private == false
  ' >/dev/null
}

manifest_is_absent() {
  local inspect_status="$1"
  local inspect_output="$2"
  ((inspect_status != 0)) && grep -Eqi '(manifest unknown|not found)' <<<"$inspect_output"
}

expected_source_revision() {
  if [[ "$GITHUB_REF" == "refs/heads/codex/single-container-publish-final-canary-0c66fdb" ]]; then
    printf '%s\n' "$CANARY_SOURCE_SHA"
    return
  fi
  printf '%s\n' "$GITHUB_SHA"
}

runnable_manifest_rows() {
  jq -er '
    def valid_digest:
      type == "string" and test("^sha256:[0-9a-f]{64}$");
    def image_manifest:
      .mediaType == "application/vnd.oci.image.manifest.v1+json"
      or .mediaType == "application/vnd.docker.distribution.manifest.v2+json";
    def image_index:
      .mediaType == "application/vnd.oci.image.index.v1+json"
      or .mediaType == "application/vnd.docker.distribution.manifest.list.v2+json";

    if image_index and (.manifests | type) == "array" then
      .manifests as $manifests
      | [
          $manifests[]
          | select(
              ((.annotations // {})["vnd.docker.reference.type"] // "")
                != "attestation-manifest"
            )
        ] as $runnable
      | [
          $manifests[]
          | select(
              ((.annotations // {})["vnd.docker.reference.type"] // "")
                == "attestation-manifest"
            )
        ] as $attestations
      | [
          $runnable[]
          | { platform: "\(.platform.os)/\(.platform.architecture)", digest }
        ] as $rows
      | if (
          ($rows | map(.platform) | sort) == ["linux/amd64", "linux/arm64"]
          and all($runnable[]; image_manifest and (.digest | valid_digest))
          and (($runnable | length) + ($attestations | length) == ($manifests | length))
          and all(
            $attestations[];
            (image_manifest)
            and (.digest | valid_digest)
            and .platform.os == "unknown"
            and .platform.architecture == "unknown"
            and (
              .annotations["vnd.docker.reference.digest"] as $subject
              | any($runnable[]; .digest == $subject)
            )
          )
        ) then
          $rows[] | "\(.platform) \(.digest)"
        else
          error("invalid runnable or attestation manifest set")
        end
    else
      error("expected a supported image index")
    end
  '
}

single_runnable_manifest_row() {
  local expected_arch="$1"
  local source_digest="$2"
  jq -er --arg expected_arch "$expected_arch" --arg source_digest "$source_digest" '
    def valid_digest:
      type == "string" and test("^sha256:[0-9a-f]{64}$");
    def image_manifest:
      .mediaType == "application/vnd.oci.image.manifest.v1+json"
      or .mediaType == "application/vnd.docker.distribution.manifest.v2+json";
    def image_index:
      .mediaType == "application/vnd.oci.image.index.v1+json"
      or .mediaType == "application/vnd.docker.distribution.manifest.list.v2+json";

    if image_manifest then
      if ($source_digest | valid_digest) then
        "linux/\($expected_arch) \($source_digest)"
      else
        error("invalid source manifest digest")
      end
    elif image_index then
      .manifests as $manifests
      | if ($manifests | type) != "array" then
          error("index is missing manifest descriptors")
        else
          [
            $manifests[]
            | select(
                ((.annotations // {})["vnd.docker.reference.type"] // "")
                  != "attestation-manifest"
              )
          ] as $runnable
          | if ($runnable | length) != 1 then
              error("expected exactly one runnable descriptor")
            else
              $runnable[0] as $image
              | [
                  $manifests[]
                  | select(
                      ((.annotations // {})["vnd.docker.reference.type"] // "")
                        == "attestation-manifest"
                    )
                ] as $attestations
              | if (
                  ($image | image_manifest)
                  and ($image.digest | valid_digest)
                  and $image.platform.os == "linux"
                  and $image.platform.architecture == $expected_arch
                  and (($runnable | length) + ($attestations | length) == ($manifests | length))
                  and all(
                    $attestations[];
                    (image_manifest)
                    and (.digest | valid_digest)
                    and .platform.os == "unknown"
                    and .platform.architecture == "unknown"
                    and .annotations["vnd.docker.reference.digest"] == $image.digest
                  )
                ) then
                  "linux/\($expected_arch) \($image.digest)"
                else
                  error("invalid runnable or attestation descriptor")
                end
            end
        end
    else
      error("unsupported manifest media type")
    end
  '
}

resolve_publication() {
  immutable_ref="${DEPLOY_IMAGE}:sha-${GITHUB_SHA}"
  release_ref=""
  release_version=""
  publish_latest=false
  publication_name="Single-container SHA image published"

  if [[ "$GITHUB_EVENT_NAME" == "workflow_dispatch" && "$GITHUB_REF" == "refs/heads/dev" ]]; then
    return
  fi

  if [[ "$GITHUB_EVENT_NAME" == "workflow_dispatch" && "$GITHUB_REF" == "refs/heads/codex/single-container-publish-final-canary-0c66fdb" ]]; then
    if [[ "$CANARY_SOURCE_SHA" != "0c66fdbf22681baa5e28566f3eb07e26c034106e" ]]; then
      echo "refusing canary for unexpected source revision: $CANARY_SOURCE_SHA" >&2
      return 1
    fi
    immutable_ref="${DEPLOY_IMAGE}:canary-sha-${CANARY_SOURCE_SHA}"
    publication_name="Single-container final PR canary published"
    return
  fi

  if [[ "$GITHUB_EVENT_NAME" == "release" && "$GITHUB_REF" == "refs/tags/${RELEASE_TAG}" ]]; then
    if [[ ! "$RELEASE_TAG" =~ ^autogpt-platform-beta-v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
      echo "refusing unsupported release tag: $RELEASE_TAG" >&2
      return 1
    fi
    release_version="v${BASH_REMATCH[1]}"
    release_ref="${DEPLOY_IMAGE}:${release_version}"
    publish_latest=true
    publication_name="Single-container release published"
    return
  fi

  echo "refusing to publish from $GITHUB_EVENT_NAME / $GITHUB_REF" >&2
  return 1
}

publication_allowed() {
  if [[ "$GITHUB_EVENT_NAME" == "workflow_dispatch" ]]; then
    if [[ "$PUBLISH_REQUESTED" != "true" ]]; then
      printf 'false\n'
      return
    fi
    if ! resolve_publication; then
      return 1
    fi
    printf 'true\n'
    return
  fi

  if [[ "$GITHUB_EVENT_NAME" == "release" ]]; then
    if [[ "$RELEASE_PRERELEASE" != "false" ]]; then
      printf 'false\n'
      return
    fi
    if [[ "$RELEASE_TAG" != autogpt-platform-beta-v* ]]; then
      printf 'false\n'
      return
    fi
    if ! resolve_publication; then
      return 1
    fi
    printf 'true\n'
    return
  fi

  printf 'false\n'
}

authorize() {
  local allowed
  allowed="$(publication_allowed)"
  if [[ "$allowed" != "true" && "$allowed" != "false" ]]; then
    echo "publication authorization returned an invalid result" >&2
    return 1
  fi
  echo "allowed=$allowed" >>"$GITHUB_OUTPUT"
}

inspect_manifest() {
  local image_ref="$1"
  docker buildx imagetools inspect "$image_ref" --format '{{json .Manifest.Digest}}' | jq -er .
}

tag_state() {
  local image_ref="$1"
  local inspect_output inspect_status

  set +e
  inspect_output="$(docker buildx imagetools inspect "$image_ref" 2>&1)"
  inspect_status=$?
  set -e
  if ((inspect_status == 0)); then
    printf 'present\n'
  elif manifest_is_absent "$inspect_status" "$inspect_output"; then
    printf 'absent\n'
  else
    echo "could not determine whether $image_ref exists" >&2
    printf '%s\n' "$inspect_output" >&2
    return 1
  fi
}

verify_manifest() {
  local image_ref="$1"
  shift
  local raw_manifest rows_output actual_rows expected_rows row platform digest image_json source_revision
  local -a rows=()

  if (($# != 2)); then
    echo "expected two smoke-tested platform descriptors" >&2
    return 1
  fi
  raw_manifest="$(docker buildx imagetools inspect --raw "$image_ref")"
  rows_output="$(runnable_manifest_rows <<<"$raw_manifest")"
  mapfile -t rows <<<"$rows_output"
  actual_rows="$(printf '%s\n' "${rows[@]}" | sort)"
  expected_rows="$(printf '%s\n' "$@" | sort)"
  if [[ "$actual_rows" != "$expected_rows" ]]; then
    echo "$image_ref does not match this run's smoke-tested platform digests" >&2
    return 1
  fi
  source_revision="$(expected_source_revision)"
  for row in "${rows[@]}"; do
    read -r platform digest <<<"$row"
    [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]
    image_json="$(
      docker buildx imagetools inspect "${DEPLOY_IMAGE}@${digest}" --format '{{json .Image}}'
    )"
    if ! jq -e --arg revision "$source_revision" '
      .config.Labels["org.opencontainers.image.revision"] == $revision
    ' <<<"$image_json" >/dev/null; then
      echo "$image_ref has an unexpected source revision for $platform" >&2
      return 1
    fi
  done
}

ensure_immutable_manifest() {
  local state resolved_digest

  state="$(tag_state "$immutable_ref")"
  if [[ "$state" == "absent" ]]; then
    docker buildx imagetools create \
      --metadata-file "$MANIFEST_METADATA" \
      --tag "$immutable_ref" \
      "${image_refs[@]}"
    manifest_digest="$(jq -er '."containerimage.descriptor".digest' "$MANIFEST_METADATA")"
  else
    manifest_digest="$(inspect_manifest "$immutable_ref")"
    echo "Reusing verified immutable tag $immutable_ref" >&2
  fi

  if [[ ! "$manifest_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "manifest publication did not return a valid sha256 digest" >&2
    return 1
  fi
  resolved_digest="$(inspect_manifest "$immutable_ref")"
  if [[ "$resolved_digest" != "$manifest_digest" ]]; then
    echo "immutable tag resolved to an unexpected digest" >&2
    return 1
  fi
  verify_manifest "$immutable_ref" "${expected_rows[@]}"
}

ensure_release_tag() {
  local manifest_digest="$1"
  local state raw_manifest

  release_manifest_digest="$manifest_digest"
  [[ -n "$release_ref" ]] || return
  state="$(tag_state "$release_ref")"
  if [[ "$state" == "absent" ]]; then
    docker buildx imagetools create \
      --annotation "index:org.opencontainers.image.version=${release_version}" \
      --tag "$release_ref" \
      "${DEPLOY_IMAGE}@${manifest_digest}"
  fi
  release_manifest_digest="$(inspect_manifest "$release_ref")"
  verify_manifest "$release_ref" "${expected_rows[@]}"
  raw_manifest="$(docker buildx imagetools inspect --raw "$release_ref")"
  if [[ "$(release_version_from_manifest <<<"$raw_manifest")" != "$release_version" ]]; then
    echo "$release_ref is missing its immutable release-version annotation" >&2
    return 1
  fi
}

release_version_from_manifest() {
  jq -er '.annotations["org.opencontainers.image.version"] | select(test("^v[0-9]+\\.[0-9]+\\.[0-9]+$"))'
}

semver_is_newer() {
  local left="${1#v}"
  local right="${2#v}"
  local left_part right_part index
  local -a left_parts right_parts

  IFS=. read -r -a left_parts <<<"$left"
  IFS=. read -r -a right_parts <<<"$right"
  ((${#left_parts[@]} == 3 && ${#right_parts[@]} == 3)) || return 2
  for index in 0 1 2; do
    left_part="${left_parts[$index]}"
    right_part="${right_parts[$index]}"
    [[ "$left_part" =~ ^[0-9]+$ && "$right_part" =~ ^[0-9]+$ ]] || return 2
    if ((10#$left_part > 10#$right_part)); then
      return 0
    fi
    if ((10#$left_part < 10#$right_part)); then
      return 1
    fi
  done
  return 1
}

latest_update_allowed() {
  local current_version="$1"
  local current_digest="$2"
  local target_version="$3"
  local target_digest="$4"
  local compare_status

  if [[ "$current_version" == "$target_version" ]]; then
    if [[ "$current_digest" != "$target_digest" ]]; then
      echo "latest already contains a different ${target_version} artifact" >&2
      return 1
    fi
    return
  fi

  set +e
  semver_is_newer "$current_version" "$target_version"
  compare_status=$?
  set -e
  if ((compare_status == 0)); then
    echo "refusing to move latest backward from $current_version to $target_version" >&2
    return 1
  fi
  if ((compare_status == 2)); then
    echo "could not compare latest release versions" >&2
    return 1
  fi
}

assert_latest_can_move() {
  local target_digest="$1"
  local state current_version current_digest raw_manifest

  state="$(tag_state "${DEPLOY_IMAGE}:latest")"
  [[ "$state" == "present" ]] || return
  raw_manifest="$(docker buildx imagetools inspect --raw "${DEPLOY_IMAGE}:latest")"
  current_version="$(release_version_from_manifest <<<"$raw_manifest")"
  current_digest="$(inspect_manifest "${DEPLOY_IMAGE}:latest")"
  latest_update_allowed "$current_version" "$current_digest" "$release_version" "$target_digest"
}

publish_latest_tag() {
  local manifest_digest="$1"
  local resolved_digest

  latest_ref="${DEPLOY_IMAGE}:latest"
  assert_latest_can_move "$manifest_digest"
  docker buildx imagetools create --tag "$latest_ref" "${DEPLOY_IMAGE}@${manifest_digest}"
  resolved_digest="$(inspect_manifest "$latest_ref")"
  if [[ "$resolved_digest" != "$manifest_digest" ]]; then
    echo "latest did not resolve to the verified manifest digest" >&2
    return 1
  fi
}

load_verified_digests() {
  local digest_file descriptor expected_arch digest_hex image_ref actual_platform raw_manifest expected_row
  local runnable_digest image_json source_revision
  local -a digest_files=()
  declare -A seen_platforms=()

  mapfile -t digest_files < <(find "$DIGEST_DIR" -maxdepth 1 -type f -print | sort)
  if ((${#digest_files[@]} != 2)); then
    echo "expected exactly two verified platform digests" >&2
    return 1
  fi

  image_refs=()
  expected_rows=()
  source_revision="$(expected_source_revision)"
  for digest_file in "${digest_files[@]}"; do
    descriptor="$(basename "$digest_file")"
    if [[ ! "$descriptor" =~ ^(amd64|arm64)-([0-9a-f]{64})$ ]]; then
      echo "invalid digest artifact name: $descriptor" >&2
      return 1
    fi
    expected_arch="${BASH_REMATCH[1]}"
    digest_hex="${BASH_REMATCH[2]}"
    if [[ -n "${seen_platforms[$expected_arch]:-}" ]]; then
      echo "duplicate digest for linux/${expected_arch}" >&2
      return 1
    fi
    seen_platforms[$expected_arch]=1
    image_ref="${DEPLOY_IMAGE}@sha256:${digest_hex}"
    actual_platform="$(
      docker buildx imagetools inspect "$image_ref" --format '{{json .Image}}' |
        platform_from_image_json "$expected_arch"
    )"
    if [[ "$actual_platform" != "linux/${expected_arch}" ]]; then
      echo "$image_ref is $actual_platform, expected linux/${expected_arch}" >&2
      return 1
    fi
    raw_manifest="$(docker buildx imagetools inspect --raw "$image_ref")"
    expected_row="$(
      single_runnable_manifest_row "$expected_arch" "sha256:${digest_hex}" <<<"$raw_manifest"
    )"
    read -r _ runnable_digest <<<"$expected_row"
    image_json="$(
      docker buildx imagetools inspect "${DEPLOY_IMAGE}@${runnable_digest}" \
        --format '{{json .Image}}'
    )"
    if ! jq -e --arg revision "$source_revision" '
      .config.Labels["org.opencontainers.image.revision"] == $revision
    ' <<<"$image_json" >/dev/null; then
      echo "$image_ref has an unexpected source revision for linux/${expected_arch}" >&2
      return 1
    fi
    image_refs+=("$image_ref")
    expected_rows+=("$expected_row")
  done
}

verify_public_repository() {
  local repository_metadata
  repository_metadata="$(
    curl --fail --silent --show-error --retry 5 --retry-all-errors \
      --connect-timeout 10 --max-time 60 \
      https://hub.docker.com/v2/repositories/significantgravitas/autogpt/
  )"
  if ! repository_is_public <<<"$repository_metadata"; then
    echo "Docker Hub repository significantgravitas/autogpt must be public" >&2
    return 1
  fi
}

write_summary() {
  local published_manifest_digest="$1"
  local sha_manifest_digest="$2"
  {
    echo "## $publication_name"
    echo
    echo "\`${immutable_ref}\`"
    echo "SHA digest: \`${sha_manifest_digest}\`"
    if [[ -n "$release_ref" ]]; then
      echo
      echo "\`${release_ref}\`"
      echo "\`${latest_ref}\`"
      echo "Release digest: \`${published_manifest_digest}\`"
    fi
    echo
    echo "Platforms: \`linux/amd64\`, \`linux/arm64\`"
  } >>"$GITHUB_STEP_SUMMARY"
}

publish() {
  local manifest_digest=""
  local release_manifest_digest=""
  local sha_manifest_digest=""
  local -a image_refs=()
  local -a expected_rows=()

  resolve_publication
  verify_public_repository
  load_verified_digests
  ensure_immutable_manifest
  sha_manifest_digest="$manifest_digest"
  ensure_release_tag "$manifest_digest"
  if [[ "$publish_latest" == true ]]; then
    publish_latest_tag "$release_manifest_digest"
    manifest_digest="$release_manifest_digest"
  fi
  write_summary "$manifest_digest" "$sha_manifest_digest"
}

assert_equal() {
  local expected="$1"
  local actual="$2"
  local message="$3"
  if [[ "$actual" != "$expected" ]]; then
    echo "$message: expected '$expected', got '$actual'" >&2
    return 1
  fi
}

self_test() {
  local actual authorization_output

  actual="$(platform_from_image_json amd64 <<<'{"os":"linux","architecture":"amd64"}')"
  assert_equal linux/amd64 "$actual" "flat image platform"
  actual="$(platform_from_image_json arm64 <<<'{"linux/arm64":{"os":"linux","architecture":"arm64"},"unknown/unknown":{}}')"
  assert_equal linux/arm64 "$actual" "platform-map image"

  repository_is_public <<<'{"namespace":"significantgravitas","name":"autogpt","is_private":false}'
  if repository_is_public <<<'{"namespace":"significantgravitas","name":"autogpt","is_private":true}'; then
    echo "private repository fixture was accepted" >&2
    return 1
  fi
  if repository_is_public <<<'{"namespace":"other","name":"autogpt","is_private":false}'; then
    echo "wrong repository fixture was accepted" >&2
    return 1
  fi

  manifest_is_absent 1 'manifest unknown'
  manifest_is_absent 1 'not found'
  if manifest_is_absent 1 'unauthorized'; then
    echo "authorization failure was treated as an absent manifest" >&2
    return 1
  fi
  if manifest_is_absent 0 ''; then
    echo "existing manifest was treated as absent" >&2
    return 1
  fi

  DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
    GITHUB_EVENT_NAME=workflow_dispatch GITHUB_REF=refs/heads/dev GITHUB_SHA=abc123 \
    RELEASE_TAG='' resolve_publication
  assert_equal docker.io/significantgravitas/autogpt:sha-abc123 "$immutable_ref" "dev immutable tag"
  assert_equal '' "$release_ref" "dev release tag"
  assert_equal false "$publish_latest" "dev latest policy"

  DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
    GITHUB_EVENT_NAME=workflow_dispatch \
    GITHUB_REF=refs/heads/codex/single-container-publish-final-canary-0c66fdb \
    GITHUB_SHA=control123 CANARY_SOURCE_SHA=0c66fdbf22681baa5e28566f3eb07e26c034106e \
    RELEASE_TAG='' resolve_publication
  assert_equal \
    docker.io/significantgravitas/autogpt:canary-sha-0c66fdbf22681baa5e28566f3eb07e26c034106e \
    "$immutable_ref" "canary immutable tag"
  assert_equal '' "$release_ref" "canary release tag"
  assert_equal false "$publish_latest" "canary latest policy"

  DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
    GITHUB_EVENT_NAME=release GITHUB_REF=refs/tags/autogpt-platform-beta-v0.7.1 \
  GITHUB_SHA=abc123 RELEASE_TAG=autogpt-platform-beta-v0.7.1 resolve_publication
  assert_equal docker.io/significantgravitas/autogpt:v0.7.1 "$release_ref" "release version tag"
  assert_equal true "$publish_latest" "release latest policy"

  actual="$(
    DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
      GITHUB_EVENT_NAME=workflow_dispatch GITHUB_REF=refs/heads/dev GITHUB_SHA=abc123 \
      PUBLISH_REQUESTED=true RELEASE_PRERELEASE='' RELEASE_TAG='' publication_allowed
  )"
  assert_equal true "$actual" "dev publication authorization"
  actual="$(
    DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
      GITHUB_EVENT_NAME=workflow_dispatch GITHUB_REF=refs/heads/dev GITHUB_SHA=abc123 \
      PUBLISH_REQUESTED=false RELEASE_PRERELEASE='' RELEASE_TAG='' publication_allowed
  )"
  assert_equal false "$actual" "validation-only dispatch authorization"
  actual="$(
    DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
      GITHUB_EVENT_NAME=release GITHUB_REF=refs/tags/autogpt-platform-beta-v0.7.1 \
      GITHUB_SHA=abc123 PUBLISH_REQUESTED='' RELEASE_PRERELEASE=false \
      RELEASE_TAG=autogpt-platform-beta-v0.7.1 publication_allowed
  )"
  assert_equal true "$actual" "release publication authorization"
  actual="$(
    DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
      GITHUB_EVENT_NAME=release GITHUB_REF=refs/tags/autogpt-platform-beta-v0.7.1 \
      GITHUB_SHA=abc123 PUBLISH_REQUESTED='' RELEASE_PRERELEASE=true \
      RELEASE_TAG=autogpt-platform-beta-v0.7.1 publication_allowed
  )"
  assert_equal false "$actual" "prerelease publication authorization"

  authorization_output="$(mktemp)"
  GITHUB_OUTPUT="$authorization_output" \
    DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
    GITHUB_EVENT_NAME=workflow_dispatch GITHUB_REF=refs/heads/dev GITHUB_SHA=abc123 \
    PUBLISH_REQUESTED=true RELEASE_PRERELEASE='' RELEASE_TAG='' authorize
  actual="$(<"$authorization_output")"
  assert_equal allowed=true "$actual" "authorization output"
  : >"$authorization_output"

  if DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
    GITHUB_EVENT_NAME=release GITHUB_REF=refs/tags/autogpt-platform-beta-v0.7 \
    GITHUB_SHA=abc123 RELEASE_TAG=autogpt-platform-beta-v0.7 resolve_publication 2>/dev/null; then
    echo "malformed release tag was accepted" >&2
    return 1
  fi
  if DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
    GITHUB_EVENT_NAME=release GITHUB_REF=refs/tags/autogpt-platform-beta-v0.7 \
    GITHUB_SHA=abc123 PUBLISH_REQUESTED='' RELEASE_PRERELEASE=false \
    RELEASE_TAG=autogpt-platform-beta-v0.7 publication_allowed 2>/dev/null; then
    echo "malformed release tag was authorized" >&2
    return 1
  fi
  if GITHUB_OUTPUT="$authorization_output" \
    DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
    GITHUB_EVENT_NAME=release GITHUB_REF=refs/tags/autogpt-platform-beta-v0.7 \
    GITHUB_SHA=abc123 PUBLISH_REQUESTED='' RELEASE_PRERELEASE=false \
    RELEASE_TAG=autogpt-platform-beta-v0.7 authorize 2>/dev/null; then
    echo "malformed release tag produced an authorization output" >&2
    return 1
  fi
  if [[ -s "$authorization_output" ]]; then
    echo "failed authorization wrote a publication output" >&2
    return 1
  fi
  rm -f "$authorization_output"
  if DEPLOY_IMAGE=docker.io/significantgravitas/autogpt \
    GITHUB_EVENT_NAME=workflow_dispatch GITHUB_REF=refs/heads/feature GITHUB_SHA=abc123 \
    RELEASE_TAG='' resolve_publication 2>/dev/null; then
    echo "feature branch publication was accepted" >&2
    return 1
  fi

  actual="$(runnable_manifest_rows <<'JSON'
{"mediaType":"application/vnd.oci.image.index.v1+json","manifests":[
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","platform":{"os":"linux","architecture":"amd64"}},
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","platform":{"os":"unknown","architecture":"unknown"},"annotations":{"vnd.docker.reference.type":"attestation-manifest","vnd.docker.reference.digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","platform":{"os":"linux","architecture":"arm64"}}
]}
JSON
  )"
  assert_equal $'linux/amd64 sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nlinux/arm64 sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc' "$actual" "attested manifest rows"

  if runnable_manifest_rows <<'JSON' >/dev/null 2>&1; then
{"mediaType":"application/vnd.oci.image.index.v1+json","manifests":[
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","platform":{"os":"linux","architecture":"amd64"}},
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","platform":{"os":"unknown","architecture":"unknown"}},
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","platform":{"os":"linux","architecture":"arm64"}}
]}
JSON
    echo "unclassified final manifest descriptor was accepted" >&2
    return 1
  fi

  actual="$(
    single_runnable_manifest_row amd64 \
      sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
      <<<'{"mediaType":"application/vnd.oci.image.manifest.v1+json"}'
  )"
  assert_equal \
    'linux/amd64 sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
    "$actual" "single image manifest row"

  actual="$(
    single_runnable_manifest_row arm64 \
      sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa <<'JSON'
{"mediaType":"application/vnd.oci.image.index.v1+json","manifests":[
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","platform":{"os":"linux","architecture":"arm64"}},
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","platform":{"os":"unknown","architecture":"unknown"},"annotations":{"vnd.docker.reference.type":"attestation-manifest","vnd.docker.reference.digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}
]}
JSON
  )"
  assert_equal \
    'linux/arm64 sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' \
    "$actual" "attested image index row"

  if single_runnable_manifest_row amd64 \
    sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
    <<<'{"mediaType":"application/vnd.oci.image.index.v1+json","manifests":[]}' \
    >/dev/null 2>&1; then
    echo "empty image index was accepted" >&2
    return 1
  fi
  if single_runnable_manifest_row amd64 \
    sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa <<'JSON' \
    >/dev/null 2>&1; then
{"mediaType":"application/vnd.oci.image.index.v1+json","manifests":[
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","platform":{"os":"linux","architecture":"amd64"}},
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","platform":{"os":"linux","architecture":"arm64"}}
]}
JSON
    echo "multi-platform source index was accepted" >&2
    return 1
  fi
  if single_runnable_manifest_row amd64 \
    sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa <<'JSON' \
    >/dev/null 2>&1; then
{"mediaType":"application/vnd.oci.image.index.v1+json","manifests":[
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","platform":{"os":"linux","architecture":"amd64"}},
  {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","platform":{"os":"unknown","architecture":"unknown"},"annotations":{"vnd.docker.reference.type":"attestation-manifest","vnd.docker.reference.digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}}
]}
JSON
    echo "unlinked attestation manifest was accepted" >&2
    return 1
  fi

  actual="$(
    release_version_from_manifest \
      <<<'{"annotations":{"org.opencontainers.image.version":"v0.7.1"}}'
  )"
  assert_equal v0.7.1 "$actual" "release-version annotation"
  if release_version_from_manifest \
    <<<'{"annotations":{"org.opencontainers.image.version":"latest"}}' >/dev/null 2>&1; then
    echo "invalid release-version annotation was accepted" >&2
    return 1
  fi

  semver_is_newer v0.7.2 v0.7.1
  semver_is_newer v1.0.0 v0.99.99
  semver_is_newer v0.08.0 v0.7.99
  if semver_is_newer v0.7.1 v0.7.1; then
    echo "equal semantic version was treated as newer" >&2
    return 1
  fi
  if semver_is_newer v0.7.0 v0.7.1; then
    echo "older semantic version was treated as newer" >&2
    return 1
  fi

  latest_update_allowed v0.7.0 sha256:old v0.7.1 sha256:new
  latest_update_allowed v0.7.1 sha256:same v0.7.1 sha256:same
  if latest_update_allowed v0.7.2 sha256:newer v0.7.1 sha256:older 2>/dev/null; then
    echo "latest rollback was accepted" >&2
    return 1
  fi
  if latest_update_allowed v0.7.1 sha256:first v0.7.1 sha256:second 2>/dev/null; then
    echo "same-version digest replacement was accepted" >&2
    return 1
  fi

  echo "single-container publication helper tests passed"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-}" in
    publish)
      publish
      ;;
    authorize)
      authorize
      ;;
    self-test)
      self_test
      ;;
    *)
      echo "usage: $0 {authorize|publish|self-test}" >&2
      exit 2
      ;;
  esac
fi
