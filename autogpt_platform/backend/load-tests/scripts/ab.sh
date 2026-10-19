#!/usr/bin/env bash
# Drives the same workload at two previews, interleaved, and prints the
# perf-compare command for the steady-phase windows it produced.
#
#   scripts/ab.sh <control PR> <candidate PR> [profile] [rounds]
#
# Control is the dev commit the change branches from, deployed with !deploy
# and nothing else changed; candidate is the change. Each round is one run at
# each preview, and rounds alternate direction (A B, B A, A B, ...) so drift
# over the session cancels out. Two rounds (A B B A) is the default, about
# 45 minutes at the baseline profile. Every run uses the same RUN_SEED.
#
# A run that fails its thresholds still counts: a slower candidate is what
# the comparison is for. Runs that covered different journeys did not send
# the same load, so when any run's coverage differs from the others' the
# script prints the coverage and no perf-compare command.
#
# The statistics are perf-compare's (.claude/skills/perf-compare): this script
# only runs the load and collects the UTC windows.
set -euo pipefail

usage="usage: scripts/ab.sh <control PR> <candidate PR> [profile] [rounds]"
control="${1:?$usage}"
candidate="${2:?$usage}"
profile="${3:-baseline}"
rounds="${4:-2}"

if ! [[ "$rounds" =~ ^[1-9][0-9]*$ ]]; then
  echo "ab.sh: rounds must be a whole number of at least 1, got '$rounds'" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

export K6_ENVIRONMENT=PREVIEW
export LOAD_PROFILE="$profile"
export RUN_SEED="${RUN_SEED:-ab-$control-$candidate}"

session="ab-$(date -u +%Y%m%dT%H%M%SZ)"
summaries=()
sides=()

run_one() {
  local side="$1" pr="$2" index="$3"
  local run_id="$session-$side$index-pr$pr"
  echo "== run $run_id: PR $pr ($side), profile $profile ==" >&2
  PR_NUMBER="$pr" RUN_ID="$run_id" ./k6.sh run --quiet tests/workload.js >&2 || true
  if [ ! -f "results/$run_id.json" ]; then
    echo "ab.sh: no summary for $run_id; stopping" >&2
    exit 1
  fi
  summaries+=("results/$run_id.json")
  sides+=("$side")
}

for ((i = 1; i <= rounds; i++)); do
  if ((i % 2 == 1)); then
    run_one A "$control" "$i"
    run_one B "$candidate" "$i"
  else
    run_one B "$candidate" "$i"
    run_one A "$control" "$i"
  fi
done

# Each journey's coverage status in one line, the same for runs that
# exercised the same journeys.
coverage() {
  python3 -c '
import json, sys
coverage = json.load(open(sys.argv[1])).get("coverage") or {}
print(" ".join(name + "=" + coverage[name]["status"] for name in sorted(coverage)))
' "$1"
}

first_coverage="$(coverage "${summaries[0]}")"
same=true
for summary in "${summaries[@]}"; do
  if [ "$(coverage "$summary")" != "$first_coverage" ]; then
    same=false
  fi
done

echo
echo "Summaries: ${summaries[*]}"
echo

if [ "$same" != true ]; then
  {
    echo "ab.sh: the runs did not cover the same journeys, so they did not send the same load:"
    for j in "${!summaries[@]}"; do
      echo "  ${sides[$j]} ${summaries[$j]}: $(coverage "${summaries[$j]}")"
    done
    echo "ab.sh: each summary's coverage section says why; fix the target and run again. No perf-compare command."
  } >&2
  exit 1
fi
case " $first_coverage" in
  *=partial* | *=skipped* | *=failed* | *=unknown*)
    echo "Every run covered the same journeys, though not all in full: $first_coverage"
    echo
    ;;
esac

windows() {
  local want="$1" out=""
  for j in "${!summaries[@]}"; do
    if [ "${sides[$j]}" = "$want" ]; then
      out+=" window=$(python3 -c '
import json, sys
w = json.load(open(sys.argv[1]))["window"]
print(w["steady_started_at"] + "/" + w["steady_ended_at"])
' "${summaries[$j]}")"
    fi
  done
  echo "$out"
}

echo "Wait a few minutes for Sentry to ingest the last events, then compare, naming the transaction you care about:"
echo
echo "S=\$(git rev-parse --show-toplevel)/.claude/skills/perf-compare/scripts/perf_compare.py"
echo "python3 \"\$S\" --project autogpt-server-previews --transaction '/api/store/skills/{slug}' \\"
echo "  --baseline env=pr-$control$(windows A) \\"
echo "  --candidate env=pr-$candidate$(windows B)"
echo
echo "For copilot turns: --transaction copilot.turn --field 'tags[copilot.turn.ttft_ms,number]'"
echo "Client-side numbers from the two sides: scripts/diff-summaries.js <control summary> <candidate summary>"
