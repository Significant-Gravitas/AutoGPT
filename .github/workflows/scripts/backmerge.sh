#!/usr/bin/env bash
# Merges master into dev whenever dev lacks it: a clean merge is pushed to dev,
# a conflicting one becomes a draft PR for a human, and `verify` is that
# human's proof that the resolution invents and loses nothing.
#
#   backmerge.sh                                       DRY_RUN=true changes nothing
#   backmerge.sh verify [<merged>] [<dev>] [<master>]
#
# Runs in a clone whose `origin` is the repository. Opening or commenting on a
# PR needs `gh`, `jq` and GITHUB_REPOSITORY.
set -euo pipefail

readonly SOURCE=master TARGET=dev
readonly BRANCH_PREFIX=backmerge/master-
readonly LABEL=conflicts-help
readonly RELEASE_TAGS='autogpt-platform-*'
readonly ATTEMPTS=3
readonly SHOWN_COMMITS=3
DRY_RUN=${DRY_RUN:-false}

main() {
  case "${1:-run}" in
    run) run ;;
    verify) shift; verify "$@" ;;
    *)
      echo "usage: $0 [run | verify [<merged>] [<dev>] [<master>]]" >&2
      exit 2
      ;;
  esac
}

run() {
  local attempt source target out status tree merged
  for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
    fetch
    source=$(git rev-parse "origin/$SOURCE")
    target=$(git rev-parse "origin/$TARGET")
    if git merge-base --is-ancestor "$source" "$target"; then
      echo "$TARGET $(short "$target") already contains $SOURCE $(short "$source"); nothing to do."
      return
    fi

    status=0
    out=$(git merge-tree --write-tree --name-only "$target" "$source") || status=$?
    case $status in
      0) ;;
      1) conflict "$source" "$target" "$out"; return ;;
      *) exit "$status" ;;
    esac
    tree=${out%%$'\n'*}

    if [[ $DRY_RUN == true ]]; then
      echo "Would push this merge of $SOURCE $(short "$source") into $TARGET $(short "$target") to $TARGET:"
      merge_message "$source" "$target" | indent '    '
      return
    fi
    # First parent dev, so dev's first-parent history stays dev's.
    merged=$(merge_message "$source" "$target" | git commit-tree "$tree" -p "$target" -p "$source")
    if git push origin "$merged:refs/heads/$TARGET"; then
      echo "Pushed $(short "$merged"), merging $SOURCE $(short "$source") into $TARGET."
      return
    fi
    fetch
    if [[ $(git rev-parse "origin/$TARGET") == "$target" ]]; then
      echo "::error title=Push to $TARGET refused::$TARGET did not move, so this token may not push to it." \
        "A repository admin adds the App this workflow runs as to the bypass list (Always allow)" \
        "of the dev-2 and dev-3 rulesets." >&2
      exit 1
    fi
    echo "$TARGET moved during the push; merging again."
  done
  echo "::error::$TARGET moved during each of $ATTEMPTS attempts; run this again." >&2
  exit 1
}

fetch() {
  git fetch --quiet origin \
    "+refs/heads/$SOURCE:refs/remotes/origin/$SOURCE" \
    "+refs/heads/$TARGET:refs/remotes/origin/$TARGET"
}

conflict() { # <source> <target> <merge-tree output>
  local source=$1 target=$2 files messages prs branch title
  files=$(sed -n '2,/^$/{/^$/!p}' <<<"$3")
  messages=$(grep '^CONFLICT' <<<"$3" || true)
  echo "Merging $SOURCE $(short "$source") into $TARGET $(short "$target") conflicts in:"
  indent '    ' <<<"$files"

  prs=$(open_backmerge_prs)
  if [[ -n $prs ]]; then
    already_open "$source" "$prs"
    return
  fi

  branch=$BRANCH_PREFIX${source:0:12}
  title="chore: back-merge master into dev ($(release "$source"))"
  if [[ $DRY_RUN == true ]]; then
    echo "Would push $(short "$source") to $branch and open a draft PR onto $TARGET: $title"
    conflict_body "$source" "$target" "$files" "$messages" "$branch" | indent '    '
    return
  fi
  if ! git ls-remote --exit-code --heads origin "$branch" >/dev/null; then
    git push origin "$source:refs/heads/$branch"
  fi
  conflict_body "$source" "$target" "$files" "$messages" "$branch" |
    gh pr create --repo "$GITHUB_REPOSITORY" --draft --base "$TARGET" --head "$branch" \
      --title "$title" --label "$LABEL" --body-file -
}

open_backmerge_prs() { # "<number> <branch>" per open back-merge PR, oldest first
  gh api --paginate --slurp "repos/$GITHUB_REPOSITORY/pulls?state=open&base=$TARGET&direction=asc&per_page=100" |
    jq -r --arg prefix "$BRANCH_PREFIX" --arg repo "$GITHUB_REPOSITORY" \
      '.[][] | select(.head.repo.full_name == $repo and (.head.ref | startswith($prefix)))
        | "\(.number) \(.head.ref)"'
}

already_open() { # <source> <open PRs>
  local source=$1 number branch marker comments
  while read -r number branch; do
    git fetch --quiet origin "+refs/heads/$branch:refs/remotes/origin/$branch"
    if git merge-base --is-ancestor "$source" "origin/$branch"; then
      echo "#$number already carries $SOURCE $(short "$source"); nothing to do."
      return
    fi
  done <<<"$2"

  read -r number branch <<<"$2"
  marker="<!-- backmerge:$source -->"
  comments=$(gh api --paginate --slurp "repos/$GITHUB_REPOSITORY/issues/$number/comments?per_page=100" | jq -r '.[][].body')
  if grep -qF "$marker" <<<"$comments"; then
    echo "#$number already says $SOURCE moved to $(short "$source"); nothing to do."
    return
  fi
  if [[ $DRY_RUN == true ]]; then
    echo "Would tell #$number that $SOURCE moved to $(short "$source")."
    return
  fi
  gh pr comment "$number" --repo "$GITHUB_REPOSITORY" --body-file - <<EOF
$marker
\`$SOURCE\` moved to $source ($(release "$source")), which this PR does not carry yet. Merge it in before landing, then re-run \`verify\`:

\`\`\`
git fetch origin $SOURCE && git merge --no-ff origin/$SOURCE
.github/workflows/scripts/backmerge.sh verify HEAD origin/$TARGET origin/$SOURCE
\`\`\`
EOF
}

conflict_body() { # <source> <target> <files> <messages> <branch>
  local source=$1 target=$2 files=$3 messages=$4 branch=$5 base file
  base=$(git merge-base "$target" "$source")
  cat <<EOF
Merging \`$SOURCE\` into \`$TARGET\` conflicts in $(wc -l <<<"$files") files, so it needs a human. This branch is \`$SOURCE\` at $source ($(release "$source")), unresolved.

**Land it by pushing the resolved head to \`$TARGET\`, never through the merge queue.** The queue squashes, which drops \`$SOURCE\`'s ancestry, so the same conflict comes back on the next push to \`$SOURCE\`. A repository admin runs \`git push origin <resolved head>:$TARGET\`, and GitHub then marks this PR merged.

### Conflicts

What each side changed since the merge base $(short "$base"). A hotfix that also landed on \`$TARGET\` as its own PR shows up on both sides.

EOF
  while read -r file; do
    echo "- \`$file\`"
    echo "  - $SOURCE: $(touched "$base" "$source" "$file")"
    echo "  - $TARGET: $(touched "$base" "$target" "$file")"
  done <<<"$files"
  cat <<EOF

\`\`\`
$messages
\`\`\`

### Resolve

\`\`\`
git fetch origin $TARGET $branch
git switch -c backmerge-resolve origin/$TARGET
git merge --no-ff origin/$branch
# resolve, then git add and git commit
.github/workflows/scripts/backmerge.sh verify HEAD origin/$TARGET origin/$branch
git push origin HEAD:$branch
\`\`\`

The push fast-forwards this branch, since its head is the merge's second parent. If \`$TARGET\` moves before landing, merge \`origin/$TARGET\` in again and re-run \`verify\`.

### Prove it

\`verify\` fails if the merge adds a line that neither side added since the merge base. It also lists, per file, the lines the merge removes that the other side did not: each needs a reason here, such as \`$TARGET\` having added a hotfix's twin and deleted it again later. Post its output on this PR before landing.
EOF
}

touched() { # <base> <tip> <path>: the commits that changed <path> since <base>
  local commits count shown
  # shellcheck disable=SC2016 # the backticks are Markdown
  commits=$(git log --format='`%h` %s' "$1..$2" -- "$3")
  if [[ -z $commits ]]; then
    echo "nothing"
    return
  fi
  count=$(wc -l <<<"$commits")
  shown=$(head -n "$SHOWN_COMMITS" <<<"$commits" | awk 'NR > 1 {printf "; "} {printf "%s", $0}')
  if ((count > SHOWN_COMMITS)); then
    shown+="; and $((count - SHOWN_COMMITS)) older"
  fi
  echo "$shown"
}

merge_message() { # <source> <target>
  echo "Merge $SOURCE into $TARGET: $(release "$1")"
  echo
  echo "Back-merges $SOURCE $1 into $TARGET. The $SOURCE commits $TARGET lacked:"
  echo
  git log --first-parent --max-count=50 --format='- %h %s' "$2..$1"
}

release() { # <sha>: the release a master commit carries, and how far past its tag
  local tag count
  if ! tag=$(git describe --tags --abbrev=0 --match "$RELEASE_TAGS" "$1" 2>/dev/null); then
    short "$1"
    return
  fi
  count=$(git rev-list --count --first-parent "$tag..$1")
  if ((count == 0)); then
    echo "$tag, $(short "$1")"
  else
    echo "$tag + $count commits, $(short "$1")"
  fi
}

verify() { # [<merged>] [<dev>] [<master>]
  local merged dev master base side failed=0
  merged=$(git rev-parse --verify "${1:-HEAD}^{commit}")
  dev=$(git rev-parse --verify "${2:-origin/$TARGET}^{commit}")
  master=$(git rev-parse --verify "${3:-origin/$SOURCE}^{commit}")
  for side in "$dev" "$master"; do
    if ! git merge-base --is-ancestor "$side" "$merged"; then
      echo "$(short "$merged") does not contain $(short "$side"); merge it in first." >&2
      exit 1
    fi
  done
  base=$(git merge-base "$dev" "$master")
  echo "merge base $(short "$base"), $TARGET $(short "$dev"), $SOURCE $(short "$master"), merged $(short "$merged")"
  echo "against $TARGET: +$(changed + "$dev" "$merged" | wc -l) -$(changed - "$dev" "$merged" | wc -l) lines;" \
    "against $SOURCE: +$(changed + "$master" "$merged" | wc -l) -$(changed - "$master" "$merged" | wc -l) lines"
  echo
  # Additions decide: a line neither side added is one the resolver wrote. A
  # removal can be a line one side added and deleted again, which no net diff shows.
  echo "Added by the merge but by neither side since the merge base (must be none):"
  unexplained_additions "$dev" "$merged" "$base" "$master" "over $TARGET" || failed=1
  unexplained_additions "$master" "$merged" "$base" "$dev" "over $SOURCE" || failed=1
  echo "Removed by the merge but not by the other side's net change (each needs a reason):"
  unexplained_removals "$dev" "$merged" "$base" "$master" "from $TARGET"
  unexplained_removals "$master" "$merged" "$base" "$dev" "from $SOURCE"
  echo
  classify "$base" "$dev" "$master" "$merged"
  return "$failed"
}

unexplained_additions() { # <from> <to> <base> <side> <label>: from->to lines not among base->side's
  # Over the whole tree: in one file, a hotfix's twin plus a later edit on the
  # same side can make a reused line look new to any net diff.
  local extra
  extra=$(LC_ALL=C comm -23 <(changed + "$1" "$2" | cut -f2- | LC_ALL=C sort) \
    <(changed + "$3" "$4" | cut -f2- | LC_ALL=C sort) && echo .)
  extra=${extra%.}
  if [[ -z $extra ]]; then
    echo "  $5: none"
    return
  fi
  echo "  $5:"
  changed + "$1" "$2" |
    awk -F'\t' 'NR == FNR {lines[$0]; next} {line = $0; sub(/^[^\t]*\t/, "", line)} line in lines' \
      <(printf '%s' "$extra") - |
    by_file
  return 1
}

unexplained_removals() { # <from> <to> <base> <side> <label>: per file, from->to removals not among base->side's
  local extra
  extra=$(LC_ALL=C comm -23 <(changed - "$1" "$2") <(changed - "$3" "$4"))
  if [[ -z $extra ]]; then
    echo "  $5: none"
    return
  fi
  echo "  $5:"
  by_file <<<"$extra"
}

by_file() { # groups "<path><TAB><line>" records under their path
  awk -F'\t' '$1 != file {file = $1; print "    " file} {sub(/^[^\t]*\t/, ""); print "      | " $0}'
}

changed() { # <+|-> <from> <to>: "<path><TAB><line>" for each added or removed line, sorted
  git diff --no-renames --no-color --no-ext-diff --no-prefix -U0 "$2" "$3" |
    awk -v sign="$1" '
      /^diff --git /{header = 1; next}
      header && /^(---|\+\+\+) / {if (substr($0, 5) != "/dev/null") path = substr($0, 5); next}
      /^@@/{header = 0; next}
      !header && substr($0, 1, 1) == sign {print path "\t" substr($0, 2)}' |
    LC_ALL=C sort
}

classify() { # <base> <dev> <master> <merged>: whose copy each file master changed carries
  local file m d x
  local -a mine=() theirs=() blend=()
  while read -r file; do
    [[ -z $file ]] && continue
    m=$(blob "$3" "$file") d=$(blob "$2" "$file") x=$(blob "$4" "$file")
    if [[ $x == "$m" ]]; then mine+=("$file")
    elif [[ $x == "$d" ]]; then theirs+=("$file")
    else blend+=("$file")
    fi
  done < <(git diff --name-only --no-renames "$1" "$3")
  echo "files $SOURCE changed: ${#mine[@]} carry $SOURCE's copy, ${#theirs[@]} $TARGET's, ${#blend[@]} a 3-way blend"
  for file in "${theirs[@]}"; do echo "    $TARGET's copy: $file"; done
  for file in "${blend[@]}"; do echo "    blend: $file"; done
}

blob() { # <rev> <path>
  git rev-parse --quiet --verify "$1:$2" || true
}

short() {
  git rev-parse --short=10 "$1"
}

indent() { # <prefix>: prefixes every line of stdin
  sed "s/^/$1/"
}

main "$@"
