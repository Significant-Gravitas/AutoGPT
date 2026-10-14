#!/usr/bin/env bash
set -euo pipefail

derived_data="${NATIVE_DERIVED_DATA:-apple/.build/SimulatorChecks}"
simulator_arch=$(uname -m)
mkdir -p e2e/artifacts

top -l 0 -s 15 -n 20 -o cpu -F -R -stats pid,command,cpu,mem,threads > e2e/artifacts/build-resources.log &
resources_pid=$!
trap 'kill "$resources_pid" 2>/dev/null || true; wait "$resources_pid" 2>/dev/null || true' EXIT

for configuration in Release Debug; do
  if ! /usr/bin/time -l xcodebuild \
    -project apple/AutoGPT.xcodeproj -scheme AutoGPT \
    -configuration "$configuration" -sdk iphonesimulator \
    -destination 'generic/platform=iOS Simulator' \
    -derivedDataPath "$derived_data" -jobs 3 -showBuildTimingSummary \
    ARCHS="$simulator_arch" ONLY_ACTIVE_ARCH=YES CODE_SIGN_IDENTITY=- \
    COMPILER_INDEX_STORE_ENABLE=NO \
    build > "e2e/artifacts/xcodebuild-$configuration.log" 2>&1; then
    tail -100 "e2e/artifacts/xcodebuild-$configuration.log"
    exit 1
  fi
done

ditto -c -k --keepParent \
  "$derived_data/Build/Products/Release-iphonesimulator/AutoGPT.app" \
  e2e/artifacts/AutoGPT-simulator.zip
