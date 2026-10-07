#!/usr/bin/env bash
set -euo pipefail

if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
  echo "This wrapper is for an isolated GitHub Actions emulator. Use run-runtime-probe.py locally." >&2
  exit 2
fi

runtime_output="${RUNNER_TEMP:?}/autogpt-android-runtime"
mkdir -p "$runtime_output"
emulator_serial="emulator-${EMULATOR_PORT:-5554}"
fixture_pid=""

cleanup() {
  adb -s "$emulator_serial" reverse --remove tcp:8765 >/dev/null 2>&1 || true
  if [[ -n "$fixture_pid" ]]; then
    kill "$fixture_pid" 2>/dev/null || true
    wait "$fixture_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

node ../testing/server.mjs > "$runtime_output/fixture.log" 2>&1 &
fixture_pid=$!
for _ in {1..20}; do
  if ! kill -0 "$fixture_pid" 2>/dev/null; then
    cat "$runtime_output/fixture.log" >&2
    exit 1
  fi
  if curl --silent --fail --max-time 1 http://127.0.0.1:8765/health >/dev/null; then
    break
  fi
  sleep 0.25
done
curl --silent --fail --max-time 2 http://127.0.0.1:8765/health > "$runtime_output/fixture-health.json"

adb -s "$emulator_serial" install -r app/build/outputs/apk/debug/app-debug.apk
adb -s "$emulator_serial" install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s "$emulator_serial" reverse tcp:8765 tcp:8765
python3 scripts/run-runtime-probe.py --serial "$emulator_serial" --disposable \
  --fixture-origin http://127.0.0.1:8765 --output "$runtime_output/probe.txt"

for screen in sign-in settings error; do
  adb -s "$emulator_serial" shell am start -W -S \
    -n com.agpt.mobile/.DesignPreviewActivity --es screen "$screen" \
    > "$runtime_output/$screen-launch.txt"
  adb -s "$emulator_serial" shell uiautomator dump \
    "/sdcard/autogpt-$screen.xml" > "$runtime_output/$screen-ui.txt"
  adb -s "$emulator_serial" exec-out screencap -p \
    > "$runtime_output/android-$screen.png"
done
