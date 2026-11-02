#!/usr/bin/env bash
set -euo pipefail

maestro_dir="$RUNNER_TEMP/native-maestro"
mkdir -p "$maestro_dir" e2e/artifacts
curl --fail --location --retry 2 \
  "https://github.com/mobile-dev-inc/Maestro/releases/download/cli-$MAESTRO_VERSION/maestro.zip" \
  -o "$maestro_dir/maestro.zip"
curl --fail --location --retry 2 \
  "https://github.com/mobile-dev-inc/Maestro/releases/download/cli-$MAESTRO_VERSION/checksums_sha256.txt" \
  -o "$maestro_dir/checksums_sha256.txt"
(cd "$maestro_dir" && shasum -a 256 -c checksums_sha256.txt && unzip -q maestro.zip)

xcrun simctl list devices available --json > e2e/artifacts/devices.json
sdk_version=$(xcrun --sdk iphonesimulator --show-sdk-version)
echo "Selecting an iPhone simulator for iOS $sdk_version"
device_id=$(python3 scripts/select-simulator.py e2e/artifacts/devices.json "$sdk_version")
echo "$device_id" > e2e/artifacts/simulator-id.txt

{
  echo "maestro=$maestro_dir/maestro/bin/maestro"
  echo "simulator-id=$device_id"
} >> "$GITHUB_OUTPUT"

xcrun simctl boot "$device_id"
xcrun simctl bootstatus "$device_id" -b
