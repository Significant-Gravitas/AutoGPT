#!/usr/bin/env bash
# Entrypoint of the Compose `db` service: the image's own, after first making
# the superuser's password on an existing database match POSTGRES_PASSWORD.
# The image applies POSTGRES_PASSWORD only when it creates the data directory,
# so without this a password generated or changed in autogpt_platform/.env
# would lock the stack out of a database created under the previous one. A
# database whose password can't be changed is not started at all: it would go
# on accepting the old password, perhaps a published one, and only that.
set -Eeo pipefail

source /usr/local/bin/docker-entrypoint.sh

# As the image's entrypoint does: options alone are options for postgres.
if [ "${1:0:1}" = '-' ]; then
  set -- postgres "$@"
fi

apply_password() {
  # The socket-only server the image uses for its own setup, so nothing can
  # connect over the network until the password is in place. Like postgres
  # started directly, it waits out crash recovery however long that takes:
  # pg_ctl gives up after a minute by default, and the container would then
  # restart into the same recovery, over and over.
  export PGCTLTIMEOUT="${PGCTLTIMEOUT:-86400}"
  if ! docker_temp_server_start "$@"; then
    # pg_ctl may have stopped waiting on a server that is still starting.
    docker_temp_server_stop >/dev/null 2>&1 || true
    return 1
  fi
  # \password hashes the password in psql, so only its SCRAM verifier reaches
  # the server, and the session keeps even that out of the logs. Without a
  # terminal, psql reads the answers to its two prompts from stdin.
  local status=0
  printf '%s\n%s\n' "$POSTGRES_PASSWORD" "$POSTGRES_PASSWORD" |
    PGPASSWORD="${PGPASSWORD:-$POSTGRES_PASSWORD}" docker_process_sql --dbname postgres \
      -c "SET log_statement = 'none'" \
      -c 'SET log_min_duration_statement = -1' \
      -c 'SET log_min_error_statement = panic' \
      -c '\password' \
      >/dev/null 2> >(grep -vE '^(Enter new password for user|Enter it again)' >&2) ||
    status=$?
  docker_temp_server_stop || status=1
  return "$status"
}

# Postgres's help, its version, or one setting (`-C name`, `-Cname`): printed
# without starting a server. Short options are read as postgres's getopt reads
# them, several to an argument, and an option's value is skipped, so that in
# `-D -Cdata` or `-Fk -Cdata`, `-Cdata` is a directory, not a setting.
prints_and_exits() {
  local arg options option
  while [ "$#" -gt 0 ]; do
    arg=$1
    shift
    case "$arg" in
      --help | --describe-config | --version) return 0 ;;
      -?*) ;;
      *) continue ;;
    esac
    options=${arg#-}
    while [ -n "$options" ]; do
      option=${options:0:1}
      options=${options:1}
      case "$option" in
        '?' | V | C) return 0 ;;
        [BcdDfhkNprStW-]) # takes a value: the rest of the argument, or the next
          [ -n "$options" ] || shift || true
          break
          ;;
      esac
    done
  done
  return 1
}

if [ "${1:-}" = postgres ] && ! prints_and_exits "$@"; then
  if ! declare -F docker_setup_env docker_create_db_directories \
    docker_temp_server_start docker_temp_server_stop docker_process_sql >/dev/null; then
    # A new data directory gets POSTGRES_PASSWORD from the image itself.
    if [ -s "${PGDATA:-/var/lib/postgresql/data}/PG_VERSION" ]; then
      echo "autogpt-db: this postgres image lacks the helpers that set" \
        "POSTGRES_PASSWORD on an existing database, so it is not started." >&2
      exit 1
    fi
  else
    docker_setup_env
    if [ "$(id -u)" = 0 ]; then
      docker_create_db_directories
      # This script as root read it, not the file again: the postgres user
      # may not be able to read a checkout's files (a umask of 077).
      exec gosu postgres bash -c "$(<"${BASH_SOURCE[0]}")" "${BASH_SOURCE[0]}" "$@"
    fi
    if [ -n "$DATABASE_ALREADY_EXISTS" ] && [ -n "$POSTGRES_PASSWORD" ] &&
      ! apply_password "$@"; then
      echo "autogpt-db: could not set POSTGRES_PASSWORD (autogpt_platform/.env) on" \
        "the existing database, so it is not started. If local connections to it" \
        "need a password, start it once with PGPASSWORD set to its current one." >&2
      exit 1
    fi
  fi
fi

exec docker-entrypoint.sh "$@"
