#!/usr/bin/env python3
"""
PR Overlap Detection Tool

Detects potential merge conflicts between a given PR and other open PRs
by checking for file overlap, line overlap, and actual merge conflicts.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChangedFile:
    path: str
    additions: list[tuple[int, int]]  # List of (start_line, end_line) ranges
    deletions: list[tuple[int, int]]
    is_rename: bool = False
    old_path: str = None


# Files that are auto-generated or rarely cause real conflicts
IGNORE_FILES = {
    "autogpt_platform/frontend/src/app/api/openapi.json",  # Auto-generated from backend
    "poetry.lock",  # Lock file, conflicts are usually trivial
    "pnpm-lock.yaml",
    "package-lock.json",
    "yarn.lock",
}


@dataclass
class PullRequest:
    number: int
    title: str
    author: str
    url: str
    head_ref: str
    base_ref: str
    files: list[str]
    changed_ranges: dict[str, ChangedFile]  # path -> ChangedFile
    updated_at: str = None  # ISO timestamp


@dataclass
class ConflictInfo:
    """Info about a single conflicting file."""
    path: str
    conflict_count: int = 0  # Number of conflict regions
    conflict_lines: int = 0  # Total lines in conflict regions
    conflict_type: str = "content"  # content, added, deleted, renamed, binary


@dataclass
class Overlap:
    pr_a: PullRequest
    pr_b: PullRequest
    overlapping_files: list[str]
    line_overlaps: dict[str, list[tuple[int, int]]]  # file -> overlapping line ranges
    has_merge_conflict: bool = False
    conflict_files: list[str] = None
    conflict_details: list[ConflictInfo] = None  # Detailed conflict info per file
    conflict_type: str = None  # None, 'pr_a_conflicts_base', 'conflict'

    def __post_init__(self):
        if self.conflict_files is None:
            self.conflict_files = []
        if self.conflict_details is None:
            self.conflict_details = []


def run_gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a gh CLI command."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"Error running gh {' '.join(args)}: {result.stderr}", file=sys.stderr)
        if check:
            sys.exit(1)
    return result


def run_git(args: list[str], cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False
    )
    if check and result.returncode != 0:
        print(f"Error running git {' '.join(args)}: {result.stderr}", file=sys.stderr)
    return result


def get_repo_info() -> tuple[str, str]:
    """Get owner and repo name from current directory or environment."""
    # Try environment first (for GitHub Actions)
    if os.environ.get("GITHUB_REPOSITORY"):
        owner, repo = os.environ["GITHUB_REPOSITORY"].split("/")
        return owner, repo
    
    # Fall back to gh repo view
    result = run_gh(["repo", "view", "--json", "owner,name"])
    data = json.loads(result.stdout)
    return data["owner"]["login"], data["name"]


def query_open_prs(owner: str, repo: str, base_branch: str) -> list[dict]:
    """Query all open PRs targeting the specified base branch."""
    prs = []
    cursor = None
    
    while True:
        after_clause = f', after: "{cursor}"' if cursor else ""
        query = f'''
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
                            baseRefName
                            files(first: 100) {{
                                nodes {{ path }}
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
        '''
        
        result = run_gh(["api", "graphql", "-f", f"query={query}"])
        data = json.loads(result.stdout)
        
        if "errors" in data:
            print(f"GraphQL errors: {data['errors']}", file=sys.stderr)
            sys.exit(1)
        
        pr_data = data["data"]["repository"]["pullRequests"]
        for edge in pr_data["edges"]:
            node = edge["node"]
            prs.append({
                "number": node["number"],
                "title": node["title"],
                "url": node["url"],
                "updated_at": node.get("updatedAt"),
                "author": node["author"]["login"] if node["author"] else "unknown",
                "head_ref": node["headRefName"],
                "base_ref": node["baseRefName"],
                "files": [f["path"] for f in node["files"]["nodes"]]
            })
        
        if not pr_data["pageInfo"]["hasNextPage"]:
            break
        cursor = pr_data["pageInfo"]["endCursor"]
    
    return prs


def parse_diff_ranges(diff: str) -> dict[str, ChangedFile]:
    """Parse a unified diff and extract changed line ranges per file."""
    files = {}
    current_file = None
    pending_rename_from = None
    pending_rename_to = None
    is_rename = False
    
    lines = diff.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect rename: "rename from path" followed by "rename to path"
        if line.startswith("rename from "):
            pending_rename_from = line[12:]
            is_rename = True
        elif line.startswith("rename to "):
            pending_rename_to = line[10:]
        
        # Also detect similarity index (indicates rename with modifications)
        elif line.startswith("similarity index"):
            is_rename = True
        
        # Match file header: +++ b/path/to/file
        elif line.startswith("+++ b/"):
            path = line[6:]
            current_file = ChangedFile(
                path=path, 
                additions=[], 
                deletions=[],
                is_rename=is_rename,
                old_path=pending_rename_from
            )
            files[path] = current_file
            # Reset rename tracking for next file
            pending_rename_from = None
            pending_rename_to = None
            is_rename = False
        
        # Match new file (--- /dev/null means new file, not rename)
        elif line.startswith("--- /dev/null"):
            is_rename = False
            pending_rename_from = None
        
        # Match hunk header: @@ -start,count +start,count @@
        elif line.startswith("@@") and current_file:
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
        
        i += 1
    
    return files


def should_ignore_file(path: str) -> bool:
    """Check if a file should be ignored for overlap detection."""
    # Check exact match
    if path in IGNORE_FILES:
        return True
    # Check if basename matches (for lock files in any directory)
    basename = path.split("/")[-1]
    if basename in IGNORE_FILES:
        return True
    return False


def get_pr_diff(pr_number: int) -> str:
    """Get the diff for a PR."""
    result = run_gh(["pr", "diff", str(pr_number)])
    return result.stdout


def ranges_overlap(range_a: tuple[int, int], range_b: tuple[int, int]) -> bool:
    """Check if two line ranges overlap."""
    return range_a[0] <= range_b[1] and range_b[0] <= range_a[1]


def find_line_overlaps(
    changes_a: dict[str, ChangedFile],
    changes_b: dict[str, ChangedFile],
    shared_files: list[str]
) -> dict[str, list[tuple[int, int]]]:
    """Find overlapping line ranges in shared files."""
    overlaps = {}
    
    for file_path in shared_files:
        # Skip ignored files
        if should_ignore_file(file_path):
            continue
            
        file_a = changes_a.get(file_path)
        file_b = changes_b.get(file_path)
        
        if not file_a or not file_b:
            continue
        
        # If either PR only renamed the file (no actual line changes), skip
        if file_a.is_rename and not file_a.additions and not file_a.deletions:
            continue
        if file_b.is_rename and not file_b.additions and not file_b.deletions:
            continue
        
        file_overlaps = []
        
        # Compare all range combinations (additions and deletions both matter)
        all_ranges_a = file_a.additions + file_a.deletions
        all_ranges_b = file_b.additions + file_b.deletions
        
        for range_a in all_ranges_a:
            for range_b in all_ranges_b:
                if ranges_overlap(range_a, range_b):
                    # Record the overlapping region
                    overlap_start = max(range_a[0], range_b[0])
                    overlap_end = min(range_a[1], range_b[1])
                    file_overlaps.append((overlap_start, overlap_end))
        
        # Deduplicate and merge overlapping ranges
        if file_overlaps:
            file_overlaps = merge_ranges(file_overlaps)
            overlaps[file_path] = file_overlaps
    
    return overlaps


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping line ranges."""
    if not ranges:
        return []
    
    # Sort by start line
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]
    
    for current in sorted_ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1:  # Overlapping or adjacent
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def classify_overlap_risk(
    overlap: "Overlap",
    changes_a: dict[str, ChangedFile],
    changes_b: dict[str, ChangedFile]
) -> str:
    """
    Classify the risk level of an overlap.
    Returns: 'conflict', 'high', 'medium', 'low'
    """
    if overlap.has_merge_conflict:
        return 'conflict'
    
    # Check if either PR involves a rename of shared files
    has_rename = False
    for file_path in overlap.overlapping_files:
        file_a = changes_a.get(file_path)
        file_b = changes_b.get(file_path)
        if (file_a and file_a.is_rename) or (file_b and file_b.is_rename):
            has_rename = True
            break
    
    if overlap.line_overlaps:
        # Count total overlapping lines
        total_overlap_lines = 0
        for ranges in overlap.line_overlaps.values():
            for start, end in ranges:
                total_overlap_lines += (end - start + 1)
        
        if total_overlap_lines > 20:
            return 'high'
        elif total_overlap_lines > 5:
            return 'medium'
        else:
            return 'low'
    
    # File overlap only (no line overlap)
    if has_rename:
        return 'medium'  # Rename + edit can cause issues
    
    return 'low'


def analyze_conflict_markers(file_path: str, cwd: str) -> ConflictInfo:
    """Analyze a conflicted file to count conflict regions and lines."""
    info = ConflictInfo(path=file_path)
    
    try:
        full_path = os.path.join(cwd, file_path)
        with open(full_path, 'r', errors='ignore') as f:
            content = f.read()
        
        lines = content.split('\n')
        in_conflict = False
        current_conflict_lines = 0
        
        for line in lines:
            if line.startswith('<<<<<<<'):
                in_conflict = True
                info.conflict_count += 1
                current_conflict_lines = 1
            elif line.startswith('>>>>>>>'):
                in_conflict = False
                current_conflict_lines += 1
                info.conflict_lines += current_conflict_lines
            elif in_conflict:
                current_conflict_lines += 1
    except:
        pass
    
    return info


def test_merge_conflict(
    owner: str,
    repo: str,
    base_branch: str,
    pr_a: PullRequest,
    pr_b: PullRequest
) -> tuple[bool, list[str], list[ConflictInfo], str]:
    """
    Test if merging both PRs would cause a conflict.
    Returns: (has_conflict, conflict_files, conflict_details, error_type)
    error_type can be: None, 'pr_a_conflicts_base', 'conflict'
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Clone with more depth to handle merges properly
        clone_url = f"https://github.com/{owner}/{repo}.git"
        result = run_git(
            ["clone", "--depth=50", "--branch", base_branch, clone_url, tmpdir],
            check=False
        )
        if result.returncode != 0:
            print(f"Failed to clone: {result.stderr}", file=sys.stderr)
            return False, [], [], None
        
        # Configure git for commits
        run_git(["config", "user.email", "otto@agpt.co"], cwd=tmpdir, check=False)
        run_git(["config", "user.name", "Otto"], cwd=tmpdir, check=False)
        
        # Fetch both PR branches
        run_git(["fetch", "origin", f"pull/{pr_a.number}/head:pr-{pr_a.number}"], cwd=tmpdir, check=False)
        run_git(["fetch", "origin", f"pull/{pr_b.number}/head:pr-{pr_b.number}"], cwd=tmpdir, check=False)
        
        # Try merging PR A (the current PR) first
        result = run_git(["merge", "--no-commit", "--no-ff", f"pr-{pr_a.number}"], cwd=tmpdir, check=False)
        if result.returncode != 0:
            # PR A itself has conflicts with base - this is important to flag!
            status_result = run_git(["status", "--porcelain"], cwd=tmpdir, check=False)
            conflict_files = []
            conflict_details = []
            for line in status_result.stdout.split("\n"):
                if len(line) >= 3 and line[0:2] in ['UU', 'AA', 'DD', 'DU', 'UD', 'AU', 'UA']:
                    file_path = line[3:].strip()
                    conflict_files.append(file_path)
                    # Analyze conflict markers
                    info = analyze_conflict_markers(file_path, tmpdir)
                    conflict_details.append(info)
            run_git(["merge", "--abort"], cwd=tmpdir, check=False)
            return True, conflict_files, conflict_details, 'pr_a_conflicts_base'
        
        # Commit the merge
        run_git(["commit", "-m", f"Merge PR #{pr_a.number}"], cwd=tmpdir, check=False)
        
        # Try merging PR B
        result = run_git(["merge", "--no-commit", "--no-ff", f"pr-{pr_b.number}"], cwd=tmpdir, check=False)
        
        if result.returncode != 0:
            # Conflict detected between A and B!
            status_result = run_git(["status", "--porcelain"], cwd=tmpdir, check=False)
            conflict_files = []
            conflict_details = []
            
            # Map git status codes to conflict types
            status_types = {
                'UU': 'content',    # Both modified
                'AA': 'both_added', # Both added
                'DD': 'both_deleted',
                'DU': 'deleted_by_us',
                'UD': 'deleted_by_them', 
                'AU': 'added_by_us',
                'UA': 'added_by_them',
            }
            
            for line in status_result.stdout.split("\n"):
                # Various conflict markers in git status
                if len(line) >= 3 and line[0:2] in status_types:
                    status_code = line[0:2]
                    file_path = line[3:].strip()
                    conflict_files.append(file_path)
                    # Analyze conflict markers
                    info = analyze_conflict_markers(file_path, tmpdir)
                    info.conflict_type = status_types.get(status_code, 'unknown')
                    conflict_details.append(info)
            
            # If no files found via status, try to get them from the merge output
            if not conflict_files and result.stderr:
                for line in result.stderr.split("\n"):
                    if "CONFLICT" in line and ":" in line:
                        # Extract file path from conflict message
                        parts = line.split(":")
                        if len(parts) > 1:
                            file_part = parts[-1].strip()
                            if file_part and not file_part.startswith("Merge"):
                                conflict_files.append(file_part)
                                conflict_details.append(ConflictInfo(path=file_part))
            
            run_git(["merge", "--abort"], cwd=tmpdir, check=False)
            return True, conflict_files, conflict_details, 'conflict'
        
        return False, [], [], None


def find_common_prefix(paths: list[str]) -> str:
    """Find the common directory prefix of a list of file paths."""
    if not paths:
        return ""
    if len(paths) == 1:
        # For single file, use the directory
        parts = paths[0].rsplit('/', 1)
        return parts[0] + '/' if len(parts) > 1 else ""
    
    # Split all paths into parts
    split_paths = [p.split('/') for p in paths]
    
    # Find common prefix parts
    common = []
    for parts in zip(*split_paths):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break
    
    return '/'.join(common) + '/' if common else ""


def format_relative_time(iso_timestamp: str) -> str:
    """Format an ISO timestamp as relative time (e.g., '2 hours ago')."""
    if not iso_timestamp:
        return ""
    
    from datetime import datetime, timezone
    try:
        # Parse ISO timestamp
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        diff = now - dt
        
        seconds = diff.total_seconds()
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        else:
            days = int(seconds / 86400)
            return f"{days}d ago"
    except:
        return ""


def format_comment(overlaps: list[Overlap], current_pr: int, changes_current: dict[str, ChangedFile], all_changes: dict[int, dict[str, ChangedFile]]) -> str:
    """Format the overlap report as a PR comment."""
    if not overlaps:
        return ""
    
    lines = ["## 🔍 PR Overlap Detection"]
    lines.append("")
    lines.append("This check compares your PR against all other open PRs targeting the same branch to detect potential merge conflicts early.")
    lines.append("")
    
    # Check if current PR conflicts with base branch
    base_conflicts = [o for o in overlaps if o.conflict_type == 'pr_a_conflicts_base']
    if base_conflicts:
        lines.append("### ⚠️ This PR has conflicts with the base branch\n")
        lines.append("Conflicts will need to be resolved before merging:\n")
        # Just show the first one since they'll all report the same base conflict
        first = base_conflicts[0]
        for f in first.conflict_files[:10]:
            lines.append(f"- `{f}`")
        if len(first.conflict_files) > 10:
            lines.append(f"- ... and {len(first.conflict_files) - 10} more files")
        lines.append("\n")
    
    # Classify each overlap
    classified = []
    for o in overlaps:
        other_pr = o.pr_b if o.pr_a.number == current_pr else o.pr_a
        other_changes = all_changes.get(other_pr.number, {})
        risk = classify_overlap_risk(o, changes_current, other_changes)
        classified.append((o, risk))
    
    # Sort by risk level
    risk_order = {'conflict': 0, 'high': 1, 'medium': 2, 'low': 3}
    classified.sort(key=lambda x: risk_order.get(x[1], 99))
    
    # Group by risk
    conflicts = [(o, r) for o, r in classified if r == 'conflict']
    high_risk = [(o, r) for o, r in classified if r == 'high']
    medium_risk = [(o, r) for o, r in classified if r == 'medium']
    low_risk = [(o, r) for o, r in classified if r == 'low']
    
    # Filter out base conflicts from the PR-to-PR conflicts
    pr_conflicts = [(o, r) for o, r in conflicts if o.conflict_type != 'pr_a_conflicts_base']
    
    if pr_conflicts:
        lines.append("### 🔴 Merge Conflicts Detected")
        lines.append("")
        lines.append("The following PRs have been tested and **will have merge conflicts** if merged after this PR. Consider coordinating with the authors.")
        lines.append("")
        for o, _ in pr_conflicts:
            other = o.pr_b if o.pr_a.number == current_pr else o.pr_a
            updated = format_relative_time(other.updated_at)
            updated_str = f" · updated {updated}" if updated else ""
            lines.append(f"- **#{other.number}** ({other.author}{updated_str}): [{other.title}]({other.url})")
            
            # Show conflict details with sizes - no truncation
            if o.conflict_details:
                all_paths = [d.path for d in o.conflict_details]
                common_prefix = find_common_prefix(all_paths)
                if common_prefix:
                    lines.append(f"  - 📁 `{common_prefix}`")
                for detail in o.conflict_details:
                    # Remove common prefix for display
                    display_path = detail.path[len(common_prefix):] if common_prefix else detail.path
                    size_str = ""
                    if detail.conflict_count > 0:
                        size_str = f" ({detail.conflict_count} conflict{'s' if detail.conflict_count > 1 else ''}, ~{detail.conflict_lines} lines)"
                    elif detail.conflict_type != 'content':
                        # Show the conflict type if no content markers found
                        type_labels = {
                            'both_added': 'added in both',
                            'both_deleted': 'deleted in both',
                            'deleted_by_us': 'deleted here, modified there',
                            'deleted_by_them': 'modified here, deleted there',
                            'added_by_us': 'added here',
                            'added_by_them': 'added there',
                        }
                        label = type_labels.get(detail.conflict_type, detail.conflict_type)
                        size_str = f" ({label})"
                    lines.append(f"    - `{display_path}`{size_str}")
            elif o.conflict_files:
                # Fallback to just file names - no truncation
                common_prefix = find_common_prefix(o.conflict_files)
                if common_prefix:
                    lines.append(f"  - 📁 `{common_prefix}`")
                for f in o.conflict_files:
                    display_path = f[len(common_prefix):] if common_prefix else f
                    lines.append(f"    - `{display_path}`")
            lines.append("")
    
    if high_risk:
        lines.append("### 🟠 High Risk — Significant Line Overlap")
        lines.append("")
        lines.append("These PRs modify many of the same lines (>20 lines). While not yet tested for conflicts, they have high potential to conflict.")
        lines.append("")
        for o, _ in high_risk:
            other = o.pr_b if o.pr_a.number == current_pr else o.pr_a
            updated = format_relative_time(other.updated_at)
            updated_str = f" · updated {updated}" if updated else ""
            lines.append(f"- **#{other.number}** ({other.author}{updated_str}): [{other.title}]({other.url})")
            all_paths = list(o.line_overlaps.keys())
            common_prefix = find_common_prefix(all_paths) if len(all_paths) > 1 else ""
            if common_prefix:
                lines.append(f"  - 📁 `{common_prefix}`")
            for file_path, ranges in o.line_overlaps.items():
                display_path = file_path[len(common_prefix):] if common_prefix else file_path
                range_strs = [f"L{r[0]}-{r[1]}" if r[0] != r[1] else f"L{r[0]}" for r in ranges]
                indent = "    " if common_prefix else "  "
                lines.append(f"{indent}- `{display_path}`: {', '.join(range_strs)}")
            lines.append("")
    
    if medium_risk:
        lines.append("### 🟡 Medium Risk — Some Line Overlap\n")
        lines.append("These PRs have some overlapping changes:\n")
        for o, _ in medium_risk:
            other = o.pr_b if o.pr_a.number == current_pr else o.pr_a
            other_changes = all_changes.get(other.number, {})
            updated = format_relative_time(other.updated_at)
            updated_str = f" · updated {updated}" if updated else ""
            lines.append(f"- **#{other.number}** ({other.author}{updated_str}): [{other.title}]({other.url})")
            
            # Note if rename is involved
            for file_path in o.overlapping_files:
                file_a = changes_current.get(file_path)
                file_b = other_changes.get(file_path)
                if (file_a and file_a.is_rename) or (file_b and file_b.is_rename):
                    lines.append(f"  - ⚠️ `{file_path}` is being renamed/moved")
                    break
            
            if o.line_overlaps:
                for file_path, ranges in o.line_overlaps.items():
                    range_strs = [f"L{r[0]}-{r[1]}" if r[0] != r[1] else f"L{r[0]}" for r in ranges]
                    lines.append(f"  - `{file_path}`: {', '.join(range_strs)}")
            else:
                non_ignored = [f for f in o.overlapping_files if not should_ignore_file(f)]
                if non_ignored:
                    lines.append(f"  - Shared files: `{'`, `'.join(non_ignored[:5])}`")
            lines.append("")
    
    if low_risk:
        lines.append("### 🟢 Low Risk — File Overlap Only\n")
        lines.append("<details><summary>These PRs touch the same files but different sections (click to expand)</summary>\n")
        for o, _ in low_risk:
            other = o.pr_b if o.pr_a.number == current_pr else o.pr_a
            non_ignored = [f for f in o.overlapping_files if not should_ignore_file(f)]
            if non_ignored:  # Only show if there are non-ignored files
                updated = format_relative_time(other.updated_at)
                updated_str = f" · updated {updated}" if updated else ""
                lines.append(f"- **#{other.number}** ({other.author}{updated_str}): [{other.title}]({other.url})")
                if o.line_overlaps:
                    for file_path, ranges in o.line_overlaps.items():
                        range_strs = [f"L{r[0]}-{r[1]}" if r[0] != r[1] else f"L{r[0]}" for r in ranges]
                        lines.append(f"  - `{file_path}`: {', '.join(range_strs)}")
                else:
                    lines.append(f"  - Shared files: `{'`, `'.join(non_ignored[:5])}`")
        lines.append("\n</details>\n")
    
    # Summary
    total = len(overlaps)
    lines.append(f"\n**Summary:** {len(conflicts)} conflicts, {len(high_risk)} high risk, {len(medium_risk)} medium risk, {len(low_risk)} low risk (out of {total} PRs with file overlap)")
    lines.append("\n---\n*Auto-generated on push. Ignores: `openapi.json`, lock files.*")
    
    return "\n".join(lines)


def post_or_update_comment(pr_number: int, body: str):
    """Post a new comment or update existing overlap detection comment."""
    # Check for existing comment
    result = run_gh(["pr", "view", str(pr_number), "--json", "comments"])
    data = json.loads(result.stdout)
    
    marker = "## 🔍 PR Overlap Detection"
    existing_comment_id = None
    
    for comment in data.get("comments", []):
        if marker in comment.get("body", ""):
            # Extract comment ID from the comment data
            # gh pr view doesn't give us the ID directly, so we need to use the API
            break
    
    # For now, just post a new comment (we can improve this later to update existing)
    if body:
        run_gh(["pr", "comment", str(pr_number), "--body", body])


def send_discord_notification(webhook_url: str, pr: PullRequest, overlaps: list[Overlap]):
    """Send a Discord notification about significant overlaps."""
    if not webhook_url or not overlaps:
        return
    
    conflicts = [o for o in overlaps if o.has_merge_conflict]
    if not conflicts:
        return  # Only notify for actual conflicts
    
    # Build Discord embed
    embed = {
        "title": f"⚠️ PR #{pr.number} has merge conflicts",
        "description": f"[{pr.title}]({pr.url})",
        "color": 0xFF0000,  # Red
        "fields": []
    }
    
    for o in conflicts:
        other = o.pr_b if o.pr_a.number == pr.number else o.pr_a
        embed["fields"].append({
            "name": f"Conflicts with #{other.number}",
            "value": f"[{other.title}]({other.url})\nFiles: `{'`, `'.join(o.conflict_files[:3])}`",
            "inline": False
        })
    
    payload = {"embeds": [embed]}
    
    # Use curl to send (avoiding extra dependencies)
    subprocess.run(
        ["curl", "-X", "POST", "-H", "Content-Type: application/json",
         "-d", json.dumps(payload), webhook_url],
        capture_output=True
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect PR overlaps and potential merge conflicts")
    parser.add_argument("pr_number", type=int, help="PR number to check")
    parser.add_argument("--base", default=None, help="Base branch (default: auto-detect from PR)")
    parser.add_argument("--skip-merge-test", action="store_true", help="Skip actual merge conflict testing")
    parser.add_argument("--discord-webhook", default=os.environ.get("DISCORD_WEBHOOK_URL"), help="Discord webhook URL for notifications")
    parser.add_argument("--dry-run", action="store_true", help="Don't post comments, just print")
    
    args = parser.parse_args()
    
    owner, repo = get_repo_info()
    print(f"Checking PR #{args.pr_number} in {owner}/{repo}")
    
    # Get current PR info
    result = run_gh(["pr", "view", str(args.pr_number), "--json", "number,title,url,author,headRefName,baseRefName,files"])
    current_pr_data = json.loads(result.stdout)
    
    base_branch = args.base or current_pr_data["baseRefName"]
    
    current_pr = PullRequest(
        number=current_pr_data["number"],
        title=current_pr_data["title"],
        author=current_pr_data["author"]["login"],
        url=current_pr_data["url"],
        head_ref=current_pr_data["headRefName"],
        base_ref=base_branch,
        files=[f["path"] for f in current_pr_data["files"]],
        changed_ranges={}
    )
    
    print(f"PR #{current_pr.number}: {current_pr.title}")
    print(f"Base branch: {base_branch}")
    print(f"Files changed: {len(current_pr.files)}")
    
    # Query other open PRs
    all_prs = query_open_prs(owner, repo, base_branch)
    other_prs = [p for p in all_prs if p["number"] != args.pr_number]
    
    print(f"Found {len(other_prs)} other open PRs targeting {base_branch}")
    
    # Find file overlaps (excluding ignored files)
    current_files = set(f for f in current_pr.files if not should_ignore_file(f))
    candidates = []
    
    for pr_data in other_prs:
        other_files = set(f for f in pr_data["files"] if not should_ignore_file(f))
        shared = current_files & other_files
        
        if shared:
            candidates.append((pr_data, list(shared)))
    
    print(f"Found {len(candidates)} PRs with file overlap (excluding ignored files)")
    
    if not candidates:
        print("No overlaps detected!")
        return
    
    # Get detailed diff for current PR
    current_diff = get_pr_diff(args.pr_number)
    current_pr.changed_ranges = parse_diff_ranges(current_diff)
    
    overlaps = []
    all_changes = {}  # Store all PR changes for risk classification
    
    for pr_data, shared_files in candidates:
        # Filter out ignored files
        non_ignored_shared = [f for f in shared_files if not should_ignore_file(f)]
        if not non_ignored_shared:
            continue  # Skip if all shared files are ignored
            
        other_pr = PullRequest(
            number=pr_data["number"],
            title=pr_data["title"],
            author=pr_data["author"],
            url=pr_data["url"],
            head_ref=pr_data["head_ref"],
            base_ref=pr_data["base_ref"],
            files=pr_data["files"],
            changed_ranges={},
            updated_at=pr_data.get("updated_at")
        )
        
        # Get diff for other PR
        other_diff = get_pr_diff(other_pr.number)
        other_pr.changed_ranges = parse_diff_ranges(other_diff)
        all_changes[other_pr.number] = other_pr.changed_ranges
        
        # Check line overlaps (now filters ignored files internally)
        line_overlaps = find_line_overlaps(
            current_pr.changed_ranges,
            other_pr.changed_ranges,
            shared_files
        )
        
        overlap = Overlap(
            pr_a=current_pr,
            pr_b=other_pr,
            overlapping_files=non_ignored_shared,
            line_overlaps=line_overlaps
        )
        
        # Test for actual merge conflicts if we have line overlaps
        if line_overlaps and not args.skip_merge_test:
            print(f"Testing merge conflict with PR #{other_pr.number}...", flush=True)
            has_conflict, conflict_files, conflict_details, error_type = test_merge_conflict(
                owner, repo, base_branch, current_pr, other_pr
            )
            overlap.has_merge_conflict = has_conflict
            overlap.conflict_files = conflict_files
            overlap.conflict_details = conflict_details
            overlap.conflict_type = error_type
        
        overlaps.append(overlap)
    
    # Generate report
    comment = format_comment(overlaps, args.pr_number, current_pr.changed_ranges, all_changes)
    
    if args.dry_run:
        print("\n" + "="*60)
        print("COMMENT PREVIEW:")
        print("="*60)
        print(comment)
    else:
        if comment:
            post_or_update_comment(args.pr_number, comment)
            print("Posted comment to PR")
        
        # Discord notification for conflicts
        if args.discord_webhook:
            send_discord_notification(args.discord_webhook, current_pr, overlaps)
    
    # Exit with non-zero if conflicts found (for CI)
    conflicts = [o for o in overlaps if o.has_merge_conflict]
    if conflicts:
        print(f"\n⚠️  Found {len(conflicts)} merge conflict(s)")
        sys.exit(1)
    
    line_overlap_count = len([o for o in overlaps if o.line_overlaps])
    if line_overlap_count:
        print(f"\n⚠️  Found {line_overlap_count} PR(s) with line overlap")
    
    print("\n✅ Done")


if __name__ == "__main__":
    main()
