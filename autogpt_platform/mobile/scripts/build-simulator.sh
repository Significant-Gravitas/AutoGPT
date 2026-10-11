#!/usr/bin/env bash
set -euo pipefail

derived_data="${NATIVE_DERIVED_DATA:-.build/SimulatorChecks}"
simulator_arch=$(uname -m)
mkdir -p e2e/artifacts

top -l 0 -s 15 -n 12 -o mem -stats pid,command,cpu,mem,threads > e2e/artifacts/build-resources.log &
resources_pid=$!
trap 'kill "$resources_pid" 2>/dev/null || true; wait "$resources_pid" 2>/dev/null || true' EXIT

if ! /usr/bin/time -l xcodebuild \
  -workspace ios/AutoGPT.xcworkspace -scheme AutoGPT \
  -configuration Release -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath "$derived_data" -jobs 3 -showBuildTimingSummary \
  ARCHS="$simulator_arch" ONLY_ACTIVE_ARCH=YES CODE_SIGN_IDENTITY=- \
  COMPILER_INDEX_STORE_ENABLE=NO GCC_OPTIMIZATION_LEVEL=0 \
  SWIFT_OPTIMIZATION_LEVEL=-Onone SWIFT_COMPILATION_MODE=wholemodule \
  DEBUG_INFORMATION_FORMAT=dwarf GCC_GENERATE_DEBUGGING_SYMBOLS=NO \
  build > e2e/artifacts/xcodebuild.log 2>&1; then
  tail -100 e2e/artifacts/xcodebuild.log
  exit 1
fi

ditto -c -k --keepParent \
  "$derived_data/Build/Products/Release-iphonesimulator/AutoGPT.app" \
  e2e/artifacts/AutoGPT-simulator.zip
