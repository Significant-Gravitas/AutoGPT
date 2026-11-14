#!/usr/bin/env bash
# Inspect imports without starting the server or relying on ldd path spellings.
set -euo pipefail

binary="${1:?expected executable, MSYS runtime directory, output directory}"
runtime="${2:?expected MSYS runtime directory}"
output="${3:?expected output directory}"
copied='|'
pending=("${binary}")

for ((index = 0; index < ${#pending[@]}; index++)); do
  imports="$(objdump -p "${pending[index]}")"
  while read -r first second library extra; do
    [[ "$first $second" == 'DLL Name:' ]] || continue
    library="${library//$'\r'/}"
    library="$(printf '%s' "$library" | tr '[:upper:]' '[:lower:]')"
    if [[ ! "$library" =~ ^[a-z0-9_.+-]+\.dll$ || -n "$extra" ]]; then
      printf 'Invalid DLL import: %s\n' "$library" >&2
      exit 1
    fi
    [[ "$library" == msys-*.dll ]] || continue
    [[ "$copied" != *"|$library|"* ]] || continue
    if [[ ! -f "${runtime}/${library}" ]]; then
      printf 'Missing MSYS runtime dependency: %s\n' "$library" >&2
      exit 1
    fi
    cp "${runtime}/${library}" "${output}/${library}"
    copied+="${library}|"
    pending+=("${output}/${library}")
    printf 'Bundled %s\n' "$library"
  done <<< "$imports"
done

if [[ "$copied" != *'|msys-2.0.dll|'* ]]; then
  printf 'Executable has no MSYS runtime import\n' >&2
  exit 1
fi
