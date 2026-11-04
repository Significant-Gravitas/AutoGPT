#!/usr/bin/env bash
# Installs the Trivy CLI from its GitHub release and puts it on PATH for the
# trivy-action steps that follow, which run with skip-setup-trivy.
#
# trivy-action's own installer downloads from get.trivy.dev. That host stopped
# resolving on 2026-11-03, which left every scan without a binary. GitHub
# serves the same release, and the archive is checked against the checksum
# that release publishes.
set -euo pipefail

version="0.70.0"
case "${RUNNER_ARCH:?}" in
  X64)
    asset="Linux-64bit"
    sha256="8b4376d5d6befe5c24d503f10ff136d9e0c49f9127a4279fd110b727929a5aa9"
    ;;
  ARM64)
    asset="Linux-ARM64"
    sha256="2f6bb988b553a1bbac6bdd1ce890f5e412439564e17522b88a4541b4f364fc8d"
    ;;
  *)
    echo "no Trivy build for runner architecture ${RUNNER_ARCH}" >&2
    exit 1
    ;;
esac

archive="${RUNNER_TEMP:?}/trivy_${version}_${asset}.tar.gz"
bin_dir="${RUNNER_TEMP}/trivy-bin"

curl --fail --silent --show-error --location --retry 3 --retry-all-errors \
  --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
  --output "$archive" \
  "https://github.com/aquasecurity/trivy/releases/download/v${version}/trivy_${version}_${asset}.tar.gz"
echo "${sha256}  ${archive}" | sha256sum --check --strict -

mkdir -p "$bin_dir"
tar -xzf "$archive" -C "$bin_dir" trivy
"$bin_dir/trivy" --version
echo "$bin_dir" >> "${GITHUB_PATH:?}"
