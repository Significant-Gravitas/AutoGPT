#!/usr/bin/env python3
"""
PR Overlap Detection Tool

Detects potential merge conflicts between a given PR and other open PRs
by checking for file overlap, line overlap, and actual merge conflicts.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional


class OverlapInfrastructureError(RuntimeError):
    """Raised when overlap detection cannot produce a complete result."""


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def parse_args(argv=None):
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect PR overlaps and potential merge conflicts"
    )
    parser.add_argument("pr_number", nargs="?", type=int, help="PR number to check")
    parser.add_argument(
        "--ref",
        dest="git_ref",
        help="Git ref to compare with the base branch without posting comments",
    )
    parser.add_argument(
        "--base", default=None, help="Base branch (default: auto-detect from PR)"
    )
    parser.add_argument(
        "--skip-merge-test",
        action="store_true",
        help="Skip actual merge conflict testing",
    )
    parser.add_argument(
        "--discord-webhook",
        default=os.environ.get("DISCORD_WEBHOOK_URL"),
        help="Discord webhook URL for notifications",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't post comments, just print"
    )

    args = parser.parse_args(argv)

    if (args.pr_number is None) == (args.git_ref is None):
        parser.error("provide exactly one of pr_number or --ref")

    if args.git_ref:
        args.dry_run = True

    return args


def main():
    """Main entry point for PR overlap detection."""
    args = parse_args()

    owner, repo = get_repo_info()
    if args.git_ref:
        base_branch = args.base or "dev"
        current_pr = fetch_ref_details(
            args.git_ref, base_branch, owner=owner, repo=repo
        )
        print(f"Checking ref {current_pr.head_ref} in {owner}/{repo}")
        print(f"Resolved commit: {current_pr.title}")
    else:
        print(f"Checking PR #{args.pr_number} in {owner}/{repo}")
        current_pr = fetch_pr_details(args.pr_number, owner=owner, repo=repo)
        base_branch = args.base or current_pr.base_ref
        print(f"PR #{current_pr.number}: {current_pr.title}")

    print(f"Base branch: {base_branch}")
    print(f"Files changed: {len(current_pr.files)}")

    # Find overlapping PRs
    overlaps, all_changes = find_overlapping_prs(
        owner, repo, base_branch, current_pr, current_pr.number, args.skip_merge_test
    )

    if not overlaps:
        print("No overlaps detected!")
        return

    # Generate and post report
    comment = format_comment(
        overlaps, current_pr.number, current_pr.changed_ranges, all_changes
    )

    if args.dry_run:
        print("\n" + "=" * 60)
        print("OVERLAP REPORT:" if args.git_ref else "COMMENT PREVIEW:")
        print("=" * 60)
        print(comment)
    else:
        if comment:
            post_or_update_comment(args.pr_number, comment)
            print("Posted comment to PR")

        if args.discord_webhook:
            send_discord_notification(args.discord_webhook, current_pr, overlaps)

    # Report results and exit
    report_results(overlaps)


# =============================================================================
# HIGH-LEVEL WORKFLOW FUNCTIONS
# =============================================================================


def fetch_pr_details(pr_number: int, owner: str, repo: str) -> "PullRequest":
    """Fetch details for a specific PR including its diff."""
    result = run_gh(
        [
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,title,url,author,headRefName,headRefOid,baseRefName,changedFiles",
        ]
    )
    data = json.loads(result.stdout)

    inventory = get_pr_files(
        owner,
        repo,
        pr_number,
        data["changedFiles"],
        expected_head_sha=data["headRefOid"],
    )
    pr = PullRequest(
        number=data["number"],
        title=data["title"],
        author=data["author"]["login"] if data.get("author") else "unknown",
        url=data["url"],
        head_ref=data["headRefName"],
        base_ref=data["baseRefName"],
        files=inventory.paths,
        changed_ranges={},
        file_aliases=inventory.aliases,
        head_sha=data["headRefOid"],
    )

    # Get detailed diff
    diff = get_pr_diff(
        pr_number,
        pr.base_ref,
        expected_head_sha=pr.head_sha,
    )
    pr.changed_ranges = parse_diff_ranges(diff)

    return pr


def fetch_ref_details(
    git_ref: str, base_branch: str, owner: str, repo: str
) -> "PullRequest":
    """Build pull-request-like details for a local Git ref."""
    resolved_ref = resolve_git_commit(git_ref)
    resolved_base = resolve_base_commit(base_branch)
    diff_range = f"{resolved_base}...{resolved_ref}"

    files_result = run_git(
        ["diff", "--name-only", "--diff-filter=ACDMRTUXB", diff_range, "--"],
        check=False,
    )
    if files_result.returncode != 0:
        print(
            f"Could not list changes for {git_ref}: {files_result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)

    diff_result = run_git(
        [
            "diff",
            "--no-ext-diff",
            "--unified=0",
            "--find-renames",
            diff_range,
            "--",
        ],
        check=False,
    )
    if diff_result.returncode != 0:
        print(
            f"Could not diff {git_ref}: {diff_result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)

    changed_ranges = parse_diff_ranges(diff_result.stdout)
    return PullRequest(
        number=0,
        title=resolved_ref,
        author="workflow_dispatch",
        url=f"https://github.com/{owner}/{repo}/commit/{resolved_ref}",
        head_ref=git_ref,
        base_ref=base_branch,
        files=[path for path in files_result.stdout.splitlines() if path],
        changed_ranges=changed_ranges,
        file_aliases=[
            change.old_path
            for change in changed_ranges.values()
            if change.old_path is not None
        ],
        head_sha=resolved_ref,
    )


def resolve_git_commit(git_ref: str) -> str:
    """Resolve a Git ref to a commit SHA or exit with a useful error."""
    result = run_git(["rev-parse", "--verify", f"{git_ref}^{{commit}}"], check=False)
    if result.returncode != 0:
        print(
            f"Could not resolve Git ref {git_ref}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return result.stdout.strip()


def resolve_base_commit(base_branch: str) -> str:
    """Resolve the base branch, preferring the remote-tracking ref."""
    for candidate in (f"origin/{base_branch}", base_branch):
        result = run_git(
            ["rev-parse", "--verify", f"{candidate}^{{commit}}"], check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()

    print(f"Could not resolve base branch {base_branch}", file=sys.stderr)
    sys.exit(1)


def find_overlapping_prs(
    owner: str,
    repo: str,
    base_branch: str,
    current_pr: "PullRequest",
    current_pr_number: int,
    skip_merge_test: bool,
) -> tuple[list["Overlap"], dict[int, dict[str, "ChangedFile"]]]:
    """Find all PRs that overlap with the current PR."""
    # Query other open PRs
    all_prs = query_open_prs(owner, repo, base_branch)
    other_prs = [
        p
        for p in all_prs
        if p["number"] != current_pr_number
        and (current_pr_number != 0 or p["head_sha"] != current_pr.title)
    ]

    print(f"Found {len(other_prs)} other open PRs targeting {base_branch}")

    # Find file overlaps (excluding ignored files, filtering by age)
    candidates = find_file_overlap_candidates(
        current_pr.files + current_pr.file_aliases,
        other_prs,
    )

    print(f"Found {len(candidates)} PRs with file overlap (excluding ignored files)")

    if not candidates:
        return [], {}

    # First pass: analyze line overlaps (no merge testing yet)
    overlaps = []
    all_changes = {}
    prs_needing_merge_test = []

    for pr_data, shared_files in candidates:
        overlap, pr_changes = analyze_pr_overlap(
            owner,
            repo,
            base_branch,
            current_pr,
            pr_data,
            shared_files,
            skip_merge_test=True,  # Always skip in first pass
        )
        if overlap:
            overlaps.append(overlap)
            all_changes[pr_data["number"]] = pr_changes
            # Track PRs that need merge testing
            if overlap.needs_merge_test and not skip_merge_test:
                prs_needing_merge_test.append(overlap)

    # Second pass: batch merge testing with shared clone
    if prs_needing_merge_test:
        run_batch_merge_tests(
            owner, repo, base_branch, current_pr, prs_needing_merge_test
        )

    return overlaps, all_changes


def run_batch_merge_tests(
    owner: str,
    repo: str,
    base_branch: str,
    current_pr: "PullRequest",
    overlaps: list["Overlap"],
):
    """Run merge tests for multiple PRs using a shared clone."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_repo(owner, repo, base_branch, tmpdir)
        configure_git(tmpdir)

        current_local_ref = "current-overlap-source"
        if current_pr.number:
            current_source = "origin"
            current_refspec = f"pull/{current_pr.number}/head:{current_local_ref}"
            current_label = f"PR #{current_pr.number}"
        else:
            current_source = require_git_success(
                run_git(["rev-parse", "--show-toplevel"], check=False),
                "resolve checked-out repository for overlap source failed",
            ).stdout.strip()
            current_refspec = f"{current_pr.title}:{current_local_ref}"
            current_label = current_pr.title

        require_git_success(
            run_git(
                ["fetch", current_source, current_refspec],
                cwd=tmpdir,
                check=False,
            ),
            f"fetch current source {current_label} failed",
        )
        verify_fetched_ref(
            tmpdir,
            current_local_ref,
            current_pr.head_sha,
            current_label,
        )

        for overlap in overlaps:
            other_pr = (
                overlap.pr_b
                if overlap.pr_a.number == current_pr.number
                else overlap.pr_a
            )
            print(f"Testing merge conflict with PR #{other_pr.number}...", flush=True)

            require_git_success(
                run_git(
                    ["checkout", "-B", "overlap-base", f"origin/{base_branch}"],
                    cwd=tmpdir,
                    check=False,
                ),
                f"reset overlap worktree to origin/{base_branch} failed",
            )
            require_git_success(
                run_git(["clean", "-fdx"], cwd=tmpdir, check=False),
                "clean overlap worktree failed",
            )

            require_git_success(
                run_git(
                    [
                        "fetch",
                        "origin",
                        f"pull/{other_pr.number}/head:pr-{other_pr.number}",
                    ],
                    cwd=tmpdir,
                    check=False,
                ),
                f"fetch PR #{other_pr.number} for merge test failed",
            )
            verify_fetched_ref(
                tmpdir,
                f"pr-{other_pr.number}",
                other_pr.head_sha,
                f"PR #{other_pr.number}",
            )

            conflict_result = try_merge_ref(
                tmpdir,
                current_local_ref,
                current_label,
            )
            if conflict_result:
                overlap.has_merge_conflict = True
                overlap.conflict_files = conflict_result[0]
                overlap.conflict_details = conflict_result[1]
                overlap.conflict_type = "pr_a_conflicts_base"
                continue

            conflict_result = try_merge_pr(
                tmpdir,
                other_pr.number,
            )
            if conflict_result:
                overlap.has_merge_conflict = True
                overlap.conflict_files = conflict_result[0]
                overlap.conflict_details = conflict_result[1]
                overlap.conflict_type = "conflict"


def analyze_pr_overlap(
    owner: str,
    repo: str,
    base_branch: str,
    current_pr: "PullRequest",
    other_pr_data: dict,
    shared_files: list[str],
    skip_merge_test: bool,
) -> tuple[Optional["Overlap"], dict[str, "ChangedFile"]]:
    """Analyze overlap between current PR and another PR."""
    # Filter out ignored files
    non_ignored_shared = [f for f in shared_files if not should_ignore_file(f)]
    if not non_ignored_shared:
        return None, {}

    other_pr = PullRequest(
        number=other_pr_data["number"],
        title=other_pr_data["title"],
        author=other_pr_data["author"],
        url=other_pr_data["url"],
        head_ref=other_pr_data["head_ref"],
        base_ref=other_pr_data["base_ref"],
        files=other_pr_data["files"],
        changed_ranges={},
        file_aliases=other_pr_data.get("file_aliases", []),
        head_sha=other_pr_data["head_sha"],
        updated_at=other_pr_data.get("updated_at"),
    )

    # Get diff for other PR
    other_diff = get_pr_diff(
        other_pr.number,
        base_branch,
        expected_head_sha=other_pr.head_sha,
    )
    other_pr.changed_ranges = parse_diff_ranges(other_diff)

    # Check line overlaps
    line_overlaps = find_line_overlaps(
        current_pr.changed_ranges, other_pr.changed_ranges, shared_files
    )

    overlap = Overlap(
        pr_a=current_pr,
        pr_b=other_pr,
        overlapping_files=non_ignored_shared,
        line_overlaps=line_overlaps,
        needs_merge_test=bool(line_overlaps)
        or requires_authoritative_merge_test(
            current_pr.changed_ranges,
            other_pr.changed_ranges,
            non_ignored_shared,
        ),
    )

    # Test for actual merge conflicts if we have line overlaps
    if overlap.needs_merge_test and not skip_merge_test:
        print(f"Testing merge conflict with PR #{other_pr.number}...", flush=True)
        (
            has_conflict,
            conflict_files,
            conflict_details,
            error_type,
        ) = test_merge_conflict(owner, repo, base_branch, current_pr, other_pr)
        overlap.has_merge_conflict = has_conflict
        overlap.conflict_files = conflict_files
        overlap.conflict_details = conflict_details
        overlap.conflict_type = error_type

    return overlap, other_pr.changed_ranges


def find_file_overlap_candidates(
    current_files: list[str], other_prs: list[dict], max_age_days: int = 14
) -> list[tuple[dict, list[str]]]:
    """Find PRs that share files with the current PR."""
    from datetime import datetime, timezone, timedelta

    current_files_set = set(f for f in current_files if not should_ignore_file(f))
    candidates = []
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    for pr_data in other_prs:
        # Filter out PRs older than max_age_days
        updated_at = pr_data.get("updated_at")
        if updated_at:
            try:
                pr_date = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                if pr_date < cutoff_date:
                    continue  # Skip old PRs
            except Exception as e:
                # If we can't parse date, include the PR (safe fallback)
                print(f"Warning: Could not parse date for PR: {e}", file=sys.stderr)

        other_files = set(
            f
            for f in pr_data["files"] + pr_data.get("file_aliases", [])
            if not should_ignore_file(f)
        )
        shared = current_files_set & other_files

        if shared:
            candidates.append((pr_data, list(shared)))

    return candidates


def report_results(overlaps: list["Overlap"]):
    """Report results (informational only, always exits 0)."""
    conflicts = [o for o in overlaps if o.has_merge_conflict]
    if conflicts:
        print(f"\n⚠️  Found {len(conflicts)} merge conflict(s)")

    line_overlap_count = len([o for o in overlaps if o.line_overlaps])
    if line_overlap_count:
        print(f"\n⚠️  Found {line_overlap_count} PR(s) with line overlap")

    print("\n✅ Done")
    # Always exit 0 - this check is informational, not a merge blocker


# =============================================================================
# COMMENT FORMATTING
# =============================================================================


def format_comment(
    overlaps: list["Overlap"],
    current_pr: int,
    changes_current: dict[str, "ChangedFile"],
    all_changes: dict[int, dict[str, "ChangedFile"]],
) -> str:
    """Format the overlap report as a PR comment."""
    if not overlaps:
        return ""

    lines = ["## 🔍 PR Overlap Detection"]
    lines.append("")
    lines.append(
        "This check compares your PR against open PRs targeting the same branch "
        "that were updated in the last 14 days."
    )
    lines.append("")

    # Check if current PR conflicts with base branch
    format_base_conflicts(overlaps, lines)

    # Classify and sort overlaps
    classified = classify_all_overlaps(
        overlaps, current_pr, changes_current, all_changes
    )

    # Group by risk
    conflicts = [(o, r) for o, r in classified if r == "conflict"]
    medium_risk = [(o, r) for o, r in classified if r == "medium"]
    low_risk = [(o, r) for o, r in classified if r == "low"]

    # Format each section
    format_conflicts_section(conflicts, current_pr, lines)
    format_medium_risk_section(
        medium_risk, current_pr, changes_current, all_changes, lines
    )
    format_low_risk_section(low_risk, current_pr, lines)

    # Summary
    total = len(overlaps)
    lines.append(
        f"\n**Summary:** {len(conflicts)} conflict(s), {len(medium_risk)} medium risk, {len(low_risk)} low risk (out of {total} PRs with file overlap)"
    )
    lines.append(
        "\n---\n*Auto-generated on push. Window: 14 days. "
        "Ignores: `openapi.json`, lock files.*"
    )

    return "\n".join(lines)


def format_base_conflicts(overlaps: list["Overlap"], lines: list[str]):
    """Format base branch conflicts section."""
    base_conflicts = [o for o in overlaps if o.conflict_type == "pr_a_conflicts_base"]
    if base_conflicts:
        lines.append("### ⚠️ This PR has conflicts with the base branch\n")
        lines.append("Conflicts will need to be resolved before merging:\n")
        first = base_conflicts[0]
        for f in first.conflict_files[:10]:
            lines.append(f"- `{f}`")
        if len(first.conflict_files) > 10:
            lines.append(f"- ... and {len(first.conflict_files) - 10} more files")
        lines.append("\n")


def format_conflicts_section(conflicts: list[tuple], current_pr: int, lines: list[str]):
    """Format the merge conflicts section."""
    pr_conflicts = [
        (o, r) for o, r in conflicts if o.conflict_type != "pr_a_conflicts_base"
    ]

    if not pr_conflicts:
        return

    lines.append("### 🔴 Merge Conflicts Detected")
    lines.append("")
    lines.append(
        "The following PRs have been tested and **will have merge conflicts** if merged after this PR. Consider coordinating with the authors."
    )
    lines.append("")

    for o, _ in pr_conflicts:
        other = o.pr_b if o.pr_a.number == current_pr else o.pr_a
        format_pr_entry(other, lines)
        format_conflict_details(o, lines)
        lines.append("")


def format_medium_risk_section(
    medium_risk: list[tuple],
    current_pr: int,
    changes_current: dict,
    all_changes: dict,
    lines: list[str],
):
    """Format the medium risk section."""
    if not medium_risk:
        return

    lines.append("### 🟡 Medium Risk — Some Line Overlap\n")
    lines.append("These PRs have some overlapping changes:\n")

    for o, _ in medium_risk:
        other = o.pr_b if o.pr_a.number == current_pr else o.pr_a
        other_changes = all_changes.get(other.number, {})
        format_pr_entry(other, lines)

        # Note if rename is involved
        for file_path in o.overlapping_files:
            file_a = resolve_changed_file(changes_current, file_path)
            file_b = resolve_changed_file(other_changes, file_path)
            if (file_a and file_a.is_rename) or (file_b and file_b.is_rename):
                lines.append(f"  - ⚠️ `{file_path}` is being renamed/moved")
                break

        if o.line_overlaps:
            for file_path, ranges in o.line_overlaps.items():
                range_strs = [
                    f"L{r[0]}-{r[1]}" if r[0] != r[1] else f"L{r[0]}" for r in ranges
                ]
                lines.append(f"  - `{file_path}`: {', '.join(range_strs)}")
        else:
            non_ignored = [f for f in o.overlapping_files if not should_ignore_file(f)]
            if non_ignored:
                lines.append(f"  - Shared files: `{'`, `'.join(non_ignored[:5])}`")
        lines.append("")


def format_low_risk_section(low_risk: list[tuple], current_pr: int, lines: list[str]):
    """Format the low risk section."""
    if not low_risk:
        return

    lines.append("### 🟢 Low Risk — File Overlap Only\n")
    lines.append(
        "<details><summary>These PRs touch the same files but different sections (click to expand)</summary>\n"
    )

    for o, _ in low_risk:
        other = o.pr_b if o.pr_a.number == current_pr else o.pr_a
        non_ignored = [f for f in o.overlapping_files if not should_ignore_file(f)]
        if non_ignored:
            format_pr_entry(other, lines)
            if o.line_overlaps:
                for file_path, ranges in o.line_overlaps.items():
                    range_strs = [
                        f"L{r[0]}-{r[1]}" if r[0] != r[1] else f"L{r[0]}"
                        for r in ranges
                    ]
                    lines.append(f"  - `{file_path}`: {', '.join(range_strs)}")
            else:
                lines.append(f"  - Shared files: `{'`, `'.join(non_ignored[:5])}`")
            lines.append("")  # Add blank line between entries

    lines.append("</details>\n")


def format_pr_entry(pr: "PullRequest", lines: list[str]):
    """Format a single PR entry line."""
    updated = format_relative_time(pr.updated_at)
    updated_str = f" · updated {updated}" if updated else ""
    # Just use #number - GitHub auto-renders it with title
    lines.append(f"- #{pr.number} ({pr.author}{updated_str})")


def format_conflict_details(overlap: "Overlap", lines: list[str]):
    """Format conflict details for a PR."""
    if overlap.conflict_details:
        all_paths = [d.path for d in overlap.conflict_details]
        common_prefix = find_common_prefix(all_paths)
        if common_prefix:
            lines.append(f"  - 📁 `{common_prefix}`")
        for detail in overlap.conflict_details:
            display_path = (
                detail.path[len(common_prefix) :] if common_prefix else detail.path
            )
            size_str = format_conflict_size(detail)
            lines.append(f"    - `{display_path}`{size_str}")
    elif overlap.conflict_files:
        common_prefix = find_common_prefix(overlap.conflict_files)
        if common_prefix:
            lines.append(f"  - 📁 `{common_prefix}`")
        for f in overlap.conflict_files:
            display_path = f[len(common_prefix) :] if common_prefix else f
            lines.append(f"    - `{display_path}`")


def format_conflict_size(detail: "ConflictInfo") -> str:
    """Format conflict size string for a file."""
    if detail.conflict_count > 0:
        return f" ({detail.conflict_count} conflict{'s' if detail.conflict_count > 1 else ''}, ~{detail.conflict_lines} lines)"
    elif detail.conflict_type != "content":
        type_labels = {
            "both_added": "added in both",
            "both_deleted": "deleted in both",
            "deleted_by_us": "deleted here, modified there",
            "deleted_by_them": "modified here, deleted there",
            "added_by_us": "added here",
            "added_by_them": "added there",
        }
        label = type_labels.get(detail.conflict_type, detail.conflict_type)
        return f" ({label})"
    return ""


def format_line_overlaps(line_overlaps: dict[str, list[tuple]], lines: list[str]):
    """Format line overlap details."""
    all_paths = list(line_overlaps.keys())
    common_prefix = find_common_prefix(all_paths) if len(all_paths) > 1 else ""
    if common_prefix:
        lines.append(f"  - 📁 `{common_prefix}`")
    for file_path, ranges in line_overlaps.items():
        display_path = file_path[len(common_prefix) :] if common_prefix else file_path
        range_strs = [f"L{r[0]}-{r[1]}" if r[0] != r[1] else f"L{r[0]}" for r in ranges]
        indent = "    " if common_prefix else "  "
        lines.append(f"{indent}- `{display_path}`: {', '.join(range_strs)}")


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================


def classify_all_overlaps(
    overlaps: list["Overlap"], current_pr: int, changes_current: dict, all_changes: dict
) -> list[tuple["Overlap", str]]:
    """Classify all overlaps by risk level and sort them."""
    classified = []
    for o in overlaps:
        other_pr = o.pr_b if o.pr_a.number == current_pr else o.pr_a
        other_changes = all_changes.get(other_pr.number, {})
        risk = classify_overlap_risk(o, changes_current, other_changes)
        classified.append((o, risk))

    def sort_key(item):
        o, risk = item
        risk_order = {"conflict": 0, "medium": 1, "low": 2}
        # For conflicts, also sort by total conflict lines (descending)
        conflict_lines = (
            sum(d.conflict_lines for d in o.conflict_details)
            if o.conflict_details
            else 0
        )
        return (risk_order.get(risk, 99), -conflict_lines)

    classified.sort(key=sort_key)

    return classified


def classify_overlap_risk(
    overlap: "Overlap",
    changes_a: dict[str, "ChangedFile"],
    changes_b: dict[str, "ChangedFile"],
) -> str:
    """Classify the risk level of an overlap."""
    if overlap.has_merge_conflict:
        return "conflict"

    has_rename = any(
        (file_a := resolve_changed_file(changes_a, file_path)) is not None
        and file_a.is_rename
        or (file_b := resolve_changed_file(changes_b, file_path)) is not None
        and file_b.is_rename
        for file_path in overlap.overlapping_files
    )

    if overlap.line_overlaps:
        total_overlap_lines = sum(
            end - start + 1
            for ranges in overlap.line_overlaps.values()
            for start, end in ranges
        )

        # Medium risk: >20 lines overlap or file rename
        if total_overlap_lines > 20 or has_rename:
            return "medium"
        else:
            return "low"

    if has_rename:
        return "medium"

    return "low"


def resolve_changed_file(
    changes: dict[str, "ChangedFile"], path: str
) -> Optional["ChangedFile"]:
    """Resolve change metadata by its current path or previous rename path."""
    direct = changes.get(path)
    if direct is not None:
        return direct
    return next(
        (change for change in changes.values() if change.old_path == path), None
    )


def find_line_overlaps(
    changes_a: dict[str, "ChangedFile"],
    changes_b: dict[str, "ChangedFile"],
    shared_files: list[str],
) -> dict[str, list[tuple[int, int]]]:
    """Find overlapping line ranges in shared files."""
    overlaps = {}

    for file_path in shared_files:
        if should_ignore_file(file_path):
            continue

        file_a = changes_a.get(file_path)
        file_b = changes_b.get(file_path)

        if not file_a or not file_b:
            continue

        # Skip pure renames
        if file_a.is_rename and not file_a.additions and not file_a.deletions:
            continue
        if file_b.is_rename and not file_b.additions and not file_b.deletions:
            continue

        # Note: This mixes old-file (deletions) and new-file (additions) line numbers,
        # which can cause false positives when PRs insert/remove many lines.
        # Acceptable for v1 since the real merge test is the authoritative check.
        file_overlaps = find_range_overlaps(
            file_a.additions + file_a.deletions, file_b.additions + file_b.deletions
        )

        if file_overlaps:
            overlaps[file_path] = merge_ranges(file_overlaps)

    return overlaps


def requires_authoritative_merge_test(
    changes_a: dict[str, "ChangedFile"],
    changes_b: dict[str, "ChangedFile"],
    shared_files: list[str],
) -> bool:
    """Return whether line ranges cannot safely rule out a merge conflict."""
    for file_path in shared_files:
        if should_ignore_file(file_path):
            continue

        file_a = changes_a.get(file_path)
        file_b = changes_b.get(file_path)
        if file_a is None or file_b is None:
            return True

        ranges_a = file_a.additions + file_a.deletions
        ranges_b = file_b.additions + file_b.deletions
        if (
            file_a.is_rename
            or file_b.is_rename
            or file_a.is_deleted
            or file_b.is_deleted
            or not ranges_a
            or not ranges_b
        ):
            return True

    return False


def find_range_overlaps(
    ranges_a: list[tuple[int, int]], ranges_b: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Find overlapping regions between two sets of ranges."""
    overlaps = []
    for range_a in ranges_a:
        for range_b in ranges_b:
            if ranges_overlap(range_a, range_b):
                overlap_start = max(range_a[0], range_b[0])
                overlap_end = min(range_a[1], range_b[1])
                overlaps.append((overlap_start, overlap_end))
    return overlaps


def ranges_overlap(range_a: tuple[int, int], range_b: tuple[int, int]) -> bool:
    """Check if two line ranges overlap."""
    return range_a[0] <= range_b[1] and range_b[0] <= range_a[1]


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping line ranges."""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]

    for current in sorted_ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


# =============================================================================
# MERGE CONFLICT TESTING
# =============================================================================


def test_merge_conflict(
    owner: str, repo: str, base_branch: str, pr_a: "PullRequest", pr_b: "PullRequest"
) -> tuple[bool, list[str], list["ConflictInfo"], str]:
    """Test if merging both PRs would cause a conflict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_repo(owner, repo, base_branch, tmpdir)
        configure_git(tmpdir)
        fetch_pr_branches(tmpdir, pr_a, pr_b)

        conflict_result = try_merge_pr(tmpdir, pr_a.number)
        if conflict_result:
            return True, conflict_result[0], conflict_result[1], "pr_a_conflicts_base"

        conflict_result = try_merge_pr(tmpdir, pr_b.number)
        if conflict_result:
            return True, conflict_result[0], conflict_result[1], "conflict"

        return False, [], [], None


def clone_repo(owner: str, repo: str, branch: str, tmpdir: str) -> None:
    """Clone the repository."""
    clone_url = f"https://github.com/{owner}/{repo}.git"
    require_git_success(
        run_git(
            [
                "clone",
                "--single-branch",
                "--branch",
                branch,
                clone_url,
                tmpdir,
            ],
            check=False,
        ),
        "clone repository failed",
    )


def configure_git(tmpdir: str):
    """Configure git for commits."""
    for args in (
        ["config", "user.email", "github-actions[bot]@users.noreply.github.com"],
        ["config", "user.name", "github-actions[bot]"],
        ["config", "commit.gpgsign", "false"],
    ):
        require_git_success(
            run_git(args, cwd=tmpdir, check=False),
            f"configure Git setting {args[1]} failed",
        )


def fetch_pr_branches(tmpdir: str, pr_a: "PullRequest", pr_b: "PullRequest") -> None:
    """Fetch both PR branches or fail when either cannot be resolved."""
    for pr in (pr_a, pr_b):
        require_git_success(
            run_git(
                ["fetch", "origin", f"pull/{pr.number}/head:pr-{pr.number}"],
                cwd=tmpdir,
                check=False,
            ),
            f"fetch PR #{pr.number} for merge test failed",
        )
        verify_fetched_ref(
            tmpdir,
            f"pr-{pr.number}",
            pr.head_sha,
            f"PR #{pr.number}",
        )


def verify_fetched_ref(
    tmpdir: str,
    git_ref: str,
    expected_head_sha: str | None,
    label: str,
) -> None:
    """Fail if a merge-test fetch no longer matches inventoried PR metadata."""
    actual_head_sha = require_git_success(
        run_git(
            ["rev-parse", "--verify", f"{git_ref}^{{commit}}"],
            cwd=tmpdir,
            check=False,
        ),
        f"resolve fetched {label} failed",
    ).stdout.strip()
    if expected_head_sha is not None and actual_head_sha != expected_head_sha:
        raise OverlapInfrastructureError(f"{label} head changed before merge testing")


def try_merge_pr(
    tmpdir: str, pr_number: int
) -> Optional[tuple[list[str], list["ConflictInfo"]]]:
    """Merge a PR, returning only genuine unmerged-index conflicts."""
    return try_merge_ref(tmpdir, f"pr-{pr_number}", f"PR #{pr_number}")


def try_merge_ref(
    tmpdir: str,
    git_ref: str,
    label: str,
) -> Optional[tuple[list[str], list["ConflictInfo"]]]:
    """Merge a ref and fail closed if Git fails without an actual conflict."""
    result = run_git(
        ["merge", "--no-ff", "--no-edit", git_ref], cwd=tmpdir, check=False
    )

    if result.returncode == 0:
        return None

    conflict_files, conflict_details = extract_conflict_info(tmpdir, result.stderr)
    if not conflict_files:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise OverlapInfrastructureError(
            f"merge {label} failed without unmerged files: {detail}"
        )
    require_git_success(
        run_git(["merge", "--abort"], cwd=tmpdir, check=False),
        f"abort conflicted merge for {label} failed",
    )

    return conflict_files, conflict_details


def extract_conflict_info(
    tmpdir: str, stderr: str
) -> tuple[list[str], list["ConflictInfo"]]:
    """Extract conflict information from Git's unmerged index."""
    unmerged_result = require_git_success(
        run_git(
            ["diff", "--name-only", "--diff-filter=U", "--"],
            cwd=tmpdir,
            check=False,
        ),
        "read unmerged index failed",
    )
    conflict_files = list(
        dict.fromkeys(path for path in unmerged_result.stdout.splitlines() if path)
    )
    if not conflict_files:
        return [], []

    status_result = require_git_success(
        run_git(["status", "--porcelain"], cwd=tmpdir, check=False),
        "read conflict status failed",
    )

    status_types = {
        "UU": "content",
        "AA": "both_added",
        "DD": "both_deleted",
        "DU": "deleted_by_us",
        "UD": "deleted_by_them",
        "AU": "added_by_us",
        "UA": "added_by_them",
    }

    conflict_details = []
    status_by_path = {
        line[3:].strip(): line[0:2]
        for line in status_result.stdout.splitlines()
        if len(line) >= 3
    }
    for file_path in conflict_files:
        info = analyze_conflict_markers(file_path, tmpdir)
        info.conflict_type = status_types.get(status_by_path.get(file_path), "unknown")
        conflict_details.append(info)

    return conflict_files, conflict_details


def analyze_conflict_markers(file_path: str, cwd: str) -> "ConflictInfo":
    """Analyze a conflicted file to count conflict regions and lines."""
    info = ConflictInfo(path=file_path)

    try:
        full_path = os.path.join(cwd, file_path)
        with open(full_path, "r", errors="ignore") as f:
            content = f.read()

        in_conflict = False
        current_conflict_lines = 0

        for line in content.split("\n"):
            if line.startswith("<<<<<<<"):
                in_conflict = True
                info.conflict_count += 1
                current_conflict_lines = 1
            elif line.startswith(">>>>>>>"):
                in_conflict = False
                current_conflict_lines += 1
                info.conflict_lines += current_conflict_lines
            elif in_conflict:
                current_conflict_lines += 1
    except Exception as e:
        print(
            f"Warning: Could not analyze conflict markers in {file_path}: {e}",
            file=sys.stderr,
        )

    return info


# =============================================================================
# DIFF PARSING
# =============================================================================


def parse_diff_ranges(diff: str) -> dict[str, "ChangedFile"]:
    """Parse a unified diff and extract changed line ranges per file."""
    files = {}
    current_file = None
    header_old_path = None

    def set_current_path(path: str) -> None:
        nonlocal current_file
        if current_file is None:
            current_file = ChangedFile(path=path, additions=[], deletions=[])
        elif current_file.path != path:
            if files.get(current_file.path) is current_file:
                del files[current_file.path]
            current_file.path = path
        files[path] = current_file

    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            current_file = None
            header_old_path = None
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = []
            if (
                len(parts) >= 4
                and parts[2].startswith("a/")
                and parts[3].startswith("b/")
            ):
                header_old_path = parts[2][2:]
                set_current_path(parts[3][2:])
        elif line.startswith("rename from "):
            if current_file is not None:
                current_file.old_path = line[12:]
                current_file.is_rename = True
        elif line.startswith("rename to "):
            set_current_path(line[10:])
            current_file.is_rename = True
            current_file.old_path = current_file.old_path or header_old_path
        elif line.startswith("similarity index"):
            if current_file is not None:
                current_file.is_rename = True
                current_file.old_path = current_file.old_path or header_old_path
        elif line.startswith("+++ b/"):
            set_current_path(line[6:])
        elif line == "+++ /dev/null" and header_old_path:
            set_current_path(header_old_path)
            current_file.is_deleted = True
        elif line.startswith("@@") and current_file:
            parse_hunk_header(line, current_file)

    return files


def parse_hunk_header(line: str, current_file: "ChangedFile"):
    """Parse a diff hunk header and add ranges to the file."""
    match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
    if match:
        old_start = int(match.group(1))
        old_count = int(match.group(2) or 1)
        new_start = int(match.group(3))
        new_count = int(match.group(4) or 1)

        if old_count > 0:
            current_file.deletions.append((old_start, old_start + old_count - 1))
        if new_count > 0:
            current_file.additions.append((new_start, new_start + new_count - 1))


# =============================================================================
# GITHUB API
# =============================================================================


def get_repo_info() -> tuple[str, str]:
    """Get owner and repo name from environment or git."""
    if os.environ.get("GITHUB_REPOSITORY"):
        owner, repo = os.environ["GITHUB_REPOSITORY"].split("/")
        return owner, repo

    result = run_gh(["repo", "view", "--json", "owner,name"])
    data = json.loads(result.stdout)
    return data["owner"]["login"], data["name"]


def query_open_prs(owner: str, repo: str, base_branch: str) -> list[dict]:
    """Query a consistent open-PR snapshot, retrying one transient page race."""
    for attempt in range(2):
        try:
            return query_open_prs_once(owner, repo, base_branch)
        except OverlapInfrastructureError as error:
            if attempt:
                raise
            print(
                f"Open PR inventory changed during pagination; retrying once: {error}",
                file=sys.stderr,
            )
    raise AssertionError("unreachable")


def query_open_prs_once(owner: str, repo: str, base_branch: str) -> list[dict]:
    """Query all open PRs targeting the specified base branch."""
    prs = []
    cursor = None
    seen_cursors = set()
    seen_pr_numbers = set()
    expected_total = None

    while True:
        after_clause = f', after: "{cursor}"' if cursor else ""
        query = f"""
        query {{
            repository(owner: "{owner}", name: "{repo}") {{
                pullRequests(
                    first: 100{after_clause},
                    states: OPEN,
                    baseRefName: "{base_branch}",
                    orderBy: {{field: UPDATED_AT, direction: DESC}}
                ) {{
                    totalCount
                    edges {{
                        node {{
                            number
                            title
                            url
                            updatedAt
                            author {{ login }}
                            headRefName
                            headRefOid
                            baseRefName
                            changedFiles
                            files(first: 100) {{
                                totalCount
                                nodes {{ path changeType }}
                                pageInfo {{ hasNextPage }}
                            }}
                        }}
                    }}
                    pageInfo {{
                        endCursor
                        hasNextPage
                    }}
                }}
            }}
        }}
        """

        result = run_gh(["api", "graphql", "-f", f"query={query}"])
        data = json.loads(result.stdout)

        if "errors" in data:
            print(f"GraphQL errors: {data['errors']}", file=sys.stderr)
            sys.exit(1)

        pr_data = data["data"]["repository"]["pullRequests"]
        page_total = pr_data["totalCount"]
        if expected_total is None:
            expected_total = page_total
        elif page_total != expected_total:
            raise OverlapInfrastructureError(
                "GraphQL open PR totalCount changed while paginating: "
                f"expected {expected_total}, got {page_total}"
            )

        for edge in pr_data["edges"]:
            node = edge["node"]
            pr_number = node["number"]
            if pr_number in seen_pr_numbers:
                raise OverlapInfrastructureError(
                    f"GraphQL returned duplicate PR #{pr_number}"
                )
            seen_pr_numbers.add(pr_number)
            inventory = get_complete_pr_files(owner, repo, node)
            prs.append(
                {
                    "number": pr_number,
                    "title": node["title"],
                    "url": node["url"],
                    "updated_at": node.get("updatedAt"),
                    "author": node["author"]["login"] if node["author"] else "unknown",
                    "head_ref": node["headRefName"],
                    "head_sha": node["headRefOid"],
                    "base_ref": node["baseRefName"],
                    "files": inventory.paths,
                    "file_aliases": inventory.aliases,
                }
            )

        if not pr_data["pageInfo"]["hasNextPage"]:
            break
        next_cursor = pr_data["pageInfo"]["endCursor"]
        if not next_cursor or next_cursor in seen_cursors:
            raise OverlapInfrastructureError(
                "GraphQL open PR pagination did not advance its cursor"
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    expected_total = expected_total or 0
    if len(prs) != expected_total:
        raise OverlapInfrastructureError(
            f"GraphQL reported {expected_total} open PRs but returned {len(prs)}"
        )

    return prs


def get_complete_pr_files(owner: str, repo: str, node: dict) -> "FileInventory":
    """Use a complete embedded file page or fall back to paginated REST."""
    pr_number = node["number"]
    expected_count = node["changedFiles"]
    files_data = node.get("files")
    embedded_paths = []
    has_rename = False
    embedded_valid = isinstance(files_data, dict)
    if embedded_valid:
        nodes = files_data.get("nodes")
        if not isinstance(nodes, list):
            embedded_valid = False
        else:
            for file_data in nodes:
                path = file_data.get("path") if isinstance(file_data, dict) else None
                if not isinstance(path, str) or not path:
                    embedded_valid = False
                    break
                embedded_paths.append(path)
                change_type = file_data.get("changeType")
                if not isinstance(change_type, str):
                    embedded_valid = False
                    break
                has_rename = has_rename or change_type == "RENAMED"

    if embedded_valid:
        page_info = files_data.get("pageInfo")
        embedded_valid = (
            files_data.get("totalCount") == expected_count
            and isinstance(page_info, dict)
            and page_info.get("hasNextPage") is False
            and len(embedded_paths) == expected_count
            and len(set(embedded_paths)) == expected_count
            and not has_rename
        )

    if embedded_valid:
        return FileInventory(paths=embedded_paths, aliases=[])
    return get_pr_files(
        owner,
        repo,
        pr_number,
        expected_count,
        expected_head_sha=node["headRefOid"],
    )


def get_pr_files(
    owner: str,
    repo: str,
    pr_number: int,
    expected_count: int,
    expected_head_sha: str | None = None,
) -> "FileInventory":
    """Fetch and validate every changed filename for a PR."""
    if not isinstance(expected_count, int) or expected_count < 0:
        raise OverlapInfrastructureError(
            f"PR #{pr_number} returned invalid changedFiles count {expected_count!r}"
        )

    files = []
    aliases = []
    seen_files = set()
    seen_aliases = set()
    page = 1
    while True:
        endpoint = (
            f"repos/{owner}/{repo}/pulls/{pr_number}/files?per_page=100&page={page}"
        )
        result = run_gh(["api", endpoint])
        try:
            page_data = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise OverlapInfrastructureError(
                f"PR #{pr_number} files page {page} was not valid JSON"
            ) from error
        if not isinstance(page_data, list):
            raise OverlapInfrastructureError(
                f"PR #{pr_number} files page {page} was not a list"
            )

        for file_data in page_data:
            filename = (
                file_data.get("filename") if isinstance(file_data, dict) else None
            )
            if not isinstance(filename, str) or not filename:
                raise OverlapInfrastructureError(
                    f"PR #{pr_number} files page {page} contained an invalid filename"
                )
            if filename in seen_files:
                raise OverlapInfrastructureError(
                    f"PR #{pr_number} returned duplicate file {filename}"
                )
            seen_files.add(filename)
            files.append(filename)
            previous_filename = file_data.get("previous_filename")
            if previous_filename is not None:
                if not isinstance(previous_filename, str) or not previous_filename:
                    raise OverlapInfrastructureError(
                        f"PR #{pr_number} files page {page} contained an invalid "
                        "previous filename"
                    )
                if (
                    previous_filename not in seen_files
                    and previous_filename not in seen_aliases
                ):
                    seen_aliases.add(previous_filename)
                    aliases.append(previous_filename)

        if len(files) > expected_count:
            break
        if len(page_data) < 100:
            break
        page += 1

    if len(files) != expected_count:
        raise OverlapInfrastructureError(
            f"PR #{pr_number} reported {expected_count} changed files "
            f"but the files API returned {len(files)}"
        )
    if expected_head_sha is not None:
        verify_pr_snapshot(
            owner,
            repo,
            pr_number,
            expected_count,
            expected_head_sha,
        )
    return FileInventory(paths=files, aliases=aliases)


def verify_pr_snapshot(
    owner: str,
    repo: str,
    pr_number: int,
    expected_count: int,
    expected_head_sha: str,
) -> None:
    """Verify a paginated REST inventory still matches its metadata snapshot."""
    result = run_gh(["api", f"repos/{owner}/{repo}/pulls/{pr_number}"])
    try:
        data = json.loads(result.stdout)
        actual_head_sha = data["head"]["sha"]
        actual_count = data["changed_files"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise OverlapInfrastructureError(
            f"PR #{pr_number} snapshot response was invalid"
        ) from error
    if actual_head_sha != expected_head_sha or actual_count != expected_count:
        raise OverlapInfrastructureError(
            f"PR #{pr_number} changed while its file inventory was being fetched"
        )


def require_git_success(
    result: subprocess.CompletedProcess,
    context: str,
) -> subprocess.CompletedProcess:
    """Return a successful Git result or raise an infrastructure error."""
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise OverlapInfrastructureError(f"{context}: {detail}")
    return result


def get_pr_diff(
    pr_number: int,
    base_branch: str,
    expected_head_sha: str | None = None,
) -> str:
    """Get a complete PR diff, falling back to checked local Git plumbing."""
    result = run_gh(["pr", "diff", str(pr_number)], check=False)
    if result.returncode == 0:
        if expected_head_sha is not None:
            verify_pr_head_sha(pr_number, expected_head_sha)
        return result.stdout

    require_git_success(
        run_git(
            [
                "fetch",
                "--no-tags",
                "origin",
                f"+refs/heads/{base_branch}:refs/remotes/origin/{base_branch}",
            ],
            check=False,
        ),
        f"refresh origin/{base_branch} for PR #{pr_number} diff failed",
    )
    require_git_success(
        run_git(
            [
                "fetch",
                "--no-tags",
                "origin",
                f"pull/{pr_number}/head",
            ],
            check=False,
        ),
        f"fetch PR #{pr_number} for diff failed",
    )
    base = require_git_success(
        run_git(
            ["rev-parse", "--verify", f"origin/{base_branch}^{{commit}}"],
            check=False,
        ),
        f"resolve origin/{base_branch} for PR #{pr_number} diff failed",
    ).stdout.strip()
    head = require_git_success(
        run_git(["rev-parse", "--verify", "FETCH_HEAD^{commit}"], check=False),
        f"resolve fetched PR #{pr_number} head failed",
    ).stdout.strip()
    if expected_head_sha is not None and head != expected_head_sha:
        raise OverlapInfrastructureError(
            f"PR #{pr_number} head changed before its diff was fetched"
        )
    merge_base = require_git_success(
        run_git(["merge-base", base, head], check=False),
        f"find merge base for PR #{pr_number} failed",
    ).stdout.strip()
    return require_git_success(
        run_git(
            [
                "diff",
                "--no-ext-diff",
                "--unified=0",
                "--find-renames",
                merge_base,
                head,
                "--",
            ],
            check=False,
        ),
        f"generate fallback diff for PR #{pr_number} failed",
    ).stdout


def verify_pr_head_sha(pr_number: int, expected_head_sha: str) -> None:
    """Verify a CLI-generated diff still corresponds to inventoried metadata."""
    result = run_gh(["pr", "view", str(pr_number), "--json", "headRefOid"])
    try:
        actual_head_sha = json.loads(result.stdout)["headRefOid"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise OverlapInfrastructureError(
            f"PR #{pr_number} head verification response was invalid"
        ) from error
    if actual_head_sha != expected_head_sha:
        raise OverlapInfrastructureError(
            f"PR #{pr_number} head changed while its diff was being fetched"
        )


def post_or_update_comment(pr_number: int, body: str):
    """Post a new comment or update existing overlap detection comment."""
    if not body:
        return

    marker = "## 🔍 PR Overlap Detection"

    # Find existing comment using GraphQL
    owner, repo = get_repo_info()
    query = f"""
    query {{
        repository(owner: "{owner}", name: "{repo}") {{
            pullRequest(number: {pr_number}) {{
                comments(first: 100) {{
                    nodes {{
                        id
                        body
                        author {{ login }}
                    }}
                }}
            }}
        }}
    }}
    """

    result = run_gh(["api", "graphql", "-f", f"query={query}"], check=False)

    existing_comment_id = None
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            comments = (
                data.get("data", {})
                .get("repository", {})
                .get("pullRequest", {})
                .get("comments", {})
                .get("nodes", [])
            )
            for comment in comments:
                if marker in comment.get("body", ""):
                    existing_comment_id = comment["id"]
                    break
        except Exception as e:
            print(
                f"Warning: Could not search for existing comment: {e}", file=sys.stderr
            )

    if existing_comment_id:
        # Update existing comment using GraphQL mutation
        # Use json.dumps for proper escaping of all special characters
        escaped_body = json.dumps(body)[1:-1]  # Strip outer quotes added by json.dumps
        mutation = f"""
        mutation {{
            updateIssueComment(input: {{id: "{existing_comment_id}", body: "{escaped_body}"}}) {{
                issueComment {{ id }}
            }}
        }}
        """
        result = run_gh(["api", "graphql", "-f", f"query={mutation}"], check=False)
        if result.returncode == 0:
            print("Updated existing overlap comment")
        else:
            # Fallback to posting new comment
            print(
                f"Failed to update comment, posting new one: {result.stderr}",
                file=sys.stderr,
            )
            run_gh(["pr", "comment", str(pr_number), "--body", body])
    else:
        # Post new comment
        run_gh(["pr", "comment", str(pr_number), "--body", body])


def send_discord_notification(
    webhook_url: str, pr: "PullRequest", overlaps: list["Overlap"]
):
    """Send a Discord notification about significant overlaps."""
    conflicts = [o for o in overlaps if o.has_merge_conflict]
    if not conflicts:
        return

    # Discord limits: max 25 fields, max 1024 chars per field value
    fields = []
    for o in conflicts[:25]:
        other = o.pr_b if o.pr_a.number == pr.number else o.pr_a
        # Build value string with truncation to stay under 1024 chars
        file_list = o.conflict_files[:3]
        files_str = f"Files: `{'`, `'.join(file_list)}`"
        if len(o.conflict_files) > 3:
            files_str += f" (+{len(o.conflict_files) - 3} more)"
        value = f"[{other.title[:100]}]({other.url})\n{files_str}"
        # Truncate if still too long
        if len(value) > 1024:
            value = value[:1020] + "..."
        fields.append(
            {"name": f"Conflicts with #{other.number}", "value": value, "inline": False}
        )

    embed = {
        "title": f"⚠️ PR #{pr.number} has merge conflicts",
        "description": f"[{pr.title}]({pr.url})",
        "color": 0xFF0000,
        "fields": fields,
    }

    if len(conflicts) > 25:
        embed["footer"] = {"text": f"... and {len(conflicts) - 25} more conflicts"}

    try:
        subprocess.run(
            [
                "curl",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "--max-time",
                "10",
                "-d",
                json.dumps({"embeds": [embed]}),
                webhook_url,
            ],
            capture_output=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        print("Warning: Discord webhook timed out", file=sys.stderr)


# =============================================================================
# UTILITIES
# =============================================================================


def run_gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a gh CLI command."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and result.returncode != 0:
        print(f"Error running gh {' '.join(args)}: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result


def run_git(
    args: list[str], cwd: str = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a git command."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise OverlapInfrastructureError(f"git {' '.join(args)} failed: {detail}")
    return result


def should_ignore_file(path: str) -> bool:
    """Check if a file should be ignored for overlap detection."""
    if path in IGNORE_FILES:
        return True
    basename = path.split("/")[-1]
    return basename in IGNORE_FILES


def find_common_prefix(paths: list[str]) -> str:
    """Find the common directory prefix of a list of file paths."""
    if not paths:
        return ""
    if len(paths) == 1:
        parts = paths[0].rsplit("/", 1)
        return parts[0] + "/" if len(parts) > 1 else ""

    split_paths = [p.split("/") for p in paths]
    common = []
    for parts in zip(*split_paths):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break

    return "/".join(common) + "/" if common else ""


def format_relative_time(iso_timestamp: str) -> str:
    """Format an ISO timestamp as relative time."""
    if not iso_timestamp:
        return ""

    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt

        seconds = diff.total_seconds()
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        else:
            return f"{int(seconds / 86400)}d ago"
    except Exception as e:
        print(f"Warning: Could not format relative time: {e}", file=sys.stderr)
        return ""


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FileInventory:
    """Canonical changed paths plus previous-filename aliases for renames."""

    paths: list[str]
    aliases: list[str]


@dataclass
class ChangedFile:
    """Represents a file changed in a PR."""

    path: str
    additions: list[tuple[int, int]]
    deletions: list[tuple[int, int]]
    is_rename: bool = False
    old_path: str = None
    is_deleted: bool = False


@dataclass
class PullRequest:
    """Represents a pull request."""

    number: int
    title: str
    author: str
    url: str
    head_ref: str
    base_ref: str
    files: list[str]
    changed_ranges: dict[str, ChangedFile]
    file_aliases: list[str] = field(default_factory=list)
    head_sha: str = None
    updated_at: str = None


@dataclass
class ConflictInfo:
    """Info about a single conflicting file."""

    path: str
    conflict_count: int = 0
    conflict_lines: int = 0
    conflict_type: str = "content"


@dataclass
class Overlap:
    """Represents an overlap between two PRs."""

    pr_a: PullRequest
    pr_b: PullRequest
    overlapping_files: list[str]
    line_overlaps: dict[str, list[tuple[int, int]]]
    needs_merge_test: bool = False
    has_merge_conflict: bool = False
    conflict_files: list[str] = None
    conflict_details: list[ConflictInfo] = None
    conflict_type: str = None

    def __post_init__(self):
        if self.conflict_files is None:
            self.conflict_files = []
        if self.conflict_details is None:
            self.conflict_details = []


# =============================================================================
# CONSTANTS
# =============================================================================

IGNORE_FILES = {
    "autogpt_platform/frontend/src/app/api/openapi.json",
    "poetry.lock",
    "pnpm-lock.yaml",
    "package-lock.json",
    "yarn.lock",
}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
