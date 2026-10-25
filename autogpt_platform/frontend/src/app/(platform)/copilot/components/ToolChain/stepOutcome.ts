// Commands whose exit 1 answers a question ("no match", "not there", "files
// differ") rather than reporting a fault.
const ANSWERS_NO = /^(?:grep|egrep|fgrep|rg|test|\[\[?|diff|cmp)(?:\s|$)/;

// Commands that only fail with a message on stderr, so a step that left
// stderr empty did not fail in one of them.
const COMPLAINS_ON_FAILURE =
  /^(?:echo|printf|true|:|cd|pwd|mkdir|cat|ls|wc|cp|mv|rm|touch|head|tail|sort|uniq|ln|chmod)(?:\s|$)/;

/** Whether a shell step failed, read from what it signalled. The tool's own
 *  error or a timeout is a failure; so is anything on stderr, since commands
 *  report their faults there (`cd: missing: No such file or directory`), and
 *  any exit code but 1 (`curl -f` exits 22). A quiet exit 1 is a normal "no"
 *  only when every command that could have produced it is a test
 *  (`grep -q TBD brief.md && echo found`); if a script could have
 *  (`grep -q x f && ./publish.sh`) it is a failure. */
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
  return !answersAQuestion(command);
}

/** Whether the quiet exit 1 of a command can only have come from a test. */
function answersAQuestion(command: string): boolean {
  const quiet = statusProducers(command).filter(
    (producer) => !COMPLAINS_ON_FAILURE.test(producer),
  );
  return (
    quiet.length > 0 && quiet.every((producer) => ANSWERS_NO.test(producer))
  );
}

/** The commands that can have set a command's failing exit status. Only the
 *  list item that ran last counts (after any unconditional `;`, newline or
 *  `&`). In its `&&`/`||` chain the status comes from the last link that ran:
 *  the final link, or an earlier one whose failure skips every link after it,
 *  which it does when only `&&` follows. A link's status is that of the last
 *  command in its pipeline. */
function statusProducers(command: string): string[] {
  const items = splitUnquoted(command, isListSeparator).filter((item) =>
    item.text.trim(),
  );
  const last = items[items.length - 1]?.text ?? "";
  const links = splitUnquoted(last, isChainSeparator);
  return links
    .filter((_, index) =>
      links.slice(index, -1).every((link) => link.separator === "&&"),
    )
    .map((link) => {
      const stages = splitUnquoted(link.text, isPipe);
      return stages[stages.length - 1].text.trim().replace(/^!\s*/, "");
    });
}

type Separator = (text: string, at: number) => number;

/** `;`, a newline or a background `&`: the next item runs whatever happened. */
function isListSeparator(text: string, at: number): number {
  const char = text[at];
  if (char === ";" || char === "\n") return 1;
  if (char !== "&" || text[at + 1] === "&" || text[at - 1] === "&") return 0;
  return isRedirect(text, at) ? 0 : 1;
}

/** `&&` or `||`: links of one chain. */
function isChainSeparator(text: string, at: number): number {
  const pair = text.slice(at, at + 2);
  return pair === "&&" || pair === "||" ? 2 : 0;
}

/** A pipe between the commands of one link (its `||` is already split off). */
function isPipe(text: string, at: number): number {
  return text[at] === "|" ? 1 : 0;
}

interface Part {
  text: string;
  // The separator that ended this part; empty for the last one.
  separator: string;
}

/** Split on separators outside quotes and not escaped by a backslash. */
function splitUnquoted(text: string, separatorAt: Separator): Part[] {
  const parts: Part[] = [];
  let current = "";
  let quote: "'" | '"' | null = null;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === "\\" && quote !== "'") {
      current += char + (text[i + 1] ?? "");
      i++;
      continue;
    }
    if (quote) {
      if (char === quote) quote = null;
      current += char;
      continue;
    }
    if (char === "'" || char === '"') {
      quote = char;
      current += char;
      continue;
    }
    const width = separatorAt(text, i);
    if (width) {
      parts.push({ text: current, separator: text.slice(i, i + width) });
      current = "";
      i += width - 1;
    } else current += char;
  }
  parts.push({ text: current, separator: "" });
  return parts;
}

/** An `&` in `2>&1`, `>&2` or `&>log` redirects output; it ends no command. */
function isRedirect(text: string, at: number): boolean {
  return /[<>]/.test(text[at - 1] ?? "") || text[at + 1] === ">";
}
