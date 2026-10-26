import { type Link, type ShellItem, shellItems } from "./shellChain";

// Commands whose exit 1 answers a question ("no match", "not there", "files
// differ") rather than reporting a fault. A test whose stderr is silenced
// still answers: the author chose to count an error as a "no".
const ANSWERS_NO = /^(?:grep|egrep|fgrep|rg|test|\[\[?|diff|cmp)(?:\s|$)/;

// Commands that only fail with a message on stderr, so a step that left
// stderr empty did not fail in one of them, unless their stderr went
// elsewhere (`2>/dev/null`, `2>&1`, `&>log`).
const COMPLAINS_ON_FAILURE =
  /^(?:echo|printf|true|:|cd|pwd|mkdir|cat|ls|wc|cp|mv|rm|touch|head|tail|sort|uniq|ln|chmod|set)(?:\s|$)/;
const SILENCES_STDERR = /(?:^|[^\w])2>|&>|>&(?!\s*\d)/;

// Past this many links the ways a chain could have run are not enumerated,
// and a quiet exit 1 is a failure.
const MAX_LINKS = 10;

// The fields that say how a shell step ended, as they appear in its JSON. A
// quoted value may hold escaped quotes; inside the command's own output the
// field names are escaped too (`\"exit_code\"`), so they never match.
const OUTCOME_FIELD =
  /"(type|stderr|exit_code|timed_out)":\s*("(?:[^"\\]|\\.)*"|-?\d+|true|false)/g;

/** What a shell step signalled. A result too large for the chat arrives as
 *  text: its JSON cut in the middle, inside a `<tool-output-truncated>`
 *  envelope. A bash result ends with stderr, exit_code and timed_out, so the
 *  cut keeps them, and the last occurrence of each is the real one. */
export function stepOutput(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value))
    return value as Record<string, unknown>;
  if (typeof value !== "string") return null;
  const fields: Record<string, unknown> = {};
  for (const [, name, raw] of value.matchAll(OUTCOME_FIELD)) {
    try {
      fields[name] = JSON.parse(raw);
    } catch {
      // A value cut mid-escape is not read; the fields after it still are.
    }
  }
  return "exit_code" in fields || fields.type === "error" ? fields : null;
}

/** Whether a shell step failed, read from what it signalled. The tool's own
 *  error or a timeout is a failure; so is anything on stderr, since commands
 *  report their faults there (`cd: missing: No such file or directory`), and
 *  any exit code but 1 (`curl -f` exits 22). A quiet exit 1 is a normal "no"
 *  only when, in every way the command could have run to it, each command
 *  that failed was a test: `grep -q TBD brief.md && echo found` is a "no",
 *  while `./build.sh || grep -q x f` means build.sh failed first. */
export function stepFailed(
  command: string | null,
  output: Record<string, unknown> | null,
): boolean {
  if (!output) return false;
  if (output.type === "error" || output.timed_out === true) return true;
  const exitCode = output.exit_code;
  if (typeof exitCode !== "number" || exitCode === 0) return false;
  const stderr = output.stderr;
  if (typeof stderr === "string" && stderr.trim()) return true;
  if (exitCode !== 1 || !command) return true;
  return !answersAQuestion(shellItems(command));
}

type Run = "no" | "fault" | "impossible";

/** Whether every way the command could have ended in a quiet exit 1 is a
 *  "no". The status comes from the last list item, or, under `set -e`, from
 *  any earlier one that failed and so ended the run. */
function answersAQuestion(items: ShellItem[]): boolean {
  const sources = items.filter(
    (item, index) => index === items.length - 1 || item.errexit,
  );
  const runs = sources.flatMap((item) => chainRuns(item.links));
  if (runs.includes("fault")) return false;
  return runs.includes("no");
}

/** Each way a chain could have run, link i failing when bit i of the mask
 *  is set. Past MAX_LINKS links they are not enumerated: a fault. */
function chainRuns(links: Link[]): Run[] {
  if (links.length === 0 || links.length > MAX_LINKS) return ["fault"];
  return Array.from({ length: 2 ** links.length }, (_, mask) =>
    judge(links, mask),
  );
}

/** One way the chain could have run. `&&` runs the next link after a
 *  success and `||` after a failure; the status is the last link's that ran.
 *  It is impossible if it ends in success or needs a command that complains
 *  to have failed without a word; a fault if a command that is not a test
 *  failed along the way, or could have earlier in a pipeline that ran: under
 *  `pipefail` its failure is the pipeline's status, and without it the
 *  commands after it read nothing (a build that failed leaves grep nothing to
 *  find). */
function judge(links: Link[], mask: number): Run {
  let failed = false;
  const verdicts: Run[] = [];
  links.forEach((link, index) => {
    const runs = link.after === "" || (link.after === "&&" ? !failed : failed);
    if (!runs) return;
    failed = Boolean(mask & (1 << index));
    const earlier = link.stages.slice(0, -1);
    if (earlier.some((stage) => quietFailure(stage) === "fault"))
      verdicts.push("fault");
    // A `!` link fails when its command succeeds, and succeeds when it fails.
    if (failed !== link.negated)
      verdicts.push(quietFailure(link.stages[link.stages.length - 1]));
  });
  if (!failed || verdicts.includes("impossible")) return "impossible";
  return verdicts.includes("fault") ? "fault" : "no";
}

/** What a command failing without a word on stderr says. */
function quietFailure(command: string): Run {
  if (ANSWERS_NO.test(command)) return "no";
  if (COMPLAINS_ON_FAILURE.test(command) && !SILENCES_STDERR.test(command))
    return "impossible";
  return "fault";
}
