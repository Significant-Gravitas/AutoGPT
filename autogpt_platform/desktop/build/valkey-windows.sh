#!/usr/bin/env bash
# Build valkey-server for Windows inside an MSYS2 shell.
#
#   valkey-windows.sh <valkey-version> <output-directory> <source-sha256>
#
# The version and the digest are the ones build/artifacts.py pins for the
# macOS build, which compiles the same source archive:
#
#   python -c "import artifacts as a; print(a.VALKEY_VERSION, a.ARTIFACTS['darwin-arm64']['valkey'].sha256)"
#
# Valkey publishes no Windows binaries and assumes POSIX throughout (fork for
# snapshots, rename over open files for the cluster config). MSYS2's runtime
# supplies those semantics; a native MSVC build was tried and cannot rename
# its own nodes.conf, which cluster mode does on every start. The result is
# valkey-server.exe plus the MSYS2 runtime DLL it needs beside it.
#
# The same approach the redis-windows project uses for Redis; Valkey is built
# here instead because it stays BSD-licensed.

set -euo pipefail
trap 'status=$?; printf "Valkey build failed at line %s (exit %s): %s\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="usage: valkey-windows.sh <valkey-version> <output-directory> <source-sha256>"
version="${1:?${usage}}"
sha256="${3:?${usage}}"
# Absolute before any cd: the build happens in a temporary directory that is
# deleted on exit, and a relative output would land inside it.
mkdir -p "${2:?${usage}}"
output="$(cd "$2" && pwd)"

pacman -S --noconfirm --needed gcc binutils make pkgconf tar >/dev/null

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
cd "${work}"
curl -fsSL -o valkey.tar.gz "https://github.com/valkey-io/valkey/archive/refs/tags/${version}.tar.gz"
echo "${sha256}  valkey.tar.gz" | sha256sum -c -
tar -xzf valkey.tar.gz
cd "valkey-${version}"

# dlfcn.h hides dladdr() behind __GNU_VISIBLE on this platform; the server's
# crash reporter uses it unconditionally.
sed -i 's/__GNU_VISIBLE/1/' /usr/include/dlfcn.h

make -j"$(nproc)" BUILD_TLS=no CFLAGS="-Wno-char-subscripts" valkey-server

cp src/valkey-server.exe "${output}/valkey-server.exe"
cp COPYING "${output}/COPYING"
printf 'Collecting Valkey runtime DLLs from PE imports\n'
bash "${script_dir}/valkey-dlls.sh" "${output}/valkey-server.exe" /usr/bin "${output}"

printf 'Checking the assembled Valkey executable\n'
"${output}/valkey-server.exe" --version
