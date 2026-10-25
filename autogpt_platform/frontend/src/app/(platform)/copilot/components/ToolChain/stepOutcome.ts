// Commands whose exit 1 answers a question ("no match", "not there", "files
// differ") rather than reporting a fault.
const ANSWERS_NO = /^(?:grep|egrep|fgrep|rg|test|\[\[?|diff|cmp)(?:\s|$)/;

/** Whether a shell step failed, read from what it signalled. The tool's own
 *  error or a timeout is a failure; so is anything on stderr, since commands
 *  report their faults there (`cd: missing: No such file or directory`), and
 *  any exit code but 1 (`curl -f` exits 22). A quiet exit 1 is a normal "no"
 *  when the command that ran last is a test-style chain (`grep -q TBD brief.md
 *  && echo found`); from anything else (a script that reports on stdout) it is
 *  a failure. */
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

/** Whether the list item that ran last (after any unconditional `;`, newline
 *  or `&`) is a chain with a predicate in it, so its exit 1 can be an answer. */
function answersAQuestion(command: string): boolean {
  const items = splitUnquoted(command, isListSeparator);
  const last = items.filter((item) => item.trim()).pop() ?? "";
  return splitUnquoted(last, isChainSeparator).some((link) =>
    ANSWERS_NO.test(link.trim().replace(/^!\s*/, "")),
  );
}

type Separator = (text: string, at: number) => number;

/** `;`, a newline or a background `&`: the next item runs whatever happened. */
function isListSeparator(text: string, at: number): number {
  const char = text[at];
  if (char === ";" || char === "\n") return 1;
  if (char !== "&" || text[at + 1] === "&" || text[at - 1] === "&") return 0;
  return isRedirect(text, at) ? 0 : 1;
}

/** `&&`, `||` or a pipe: links of one chain. */
function isChainSeparator(text: string, at: number): number {
  const pair = text.slice(at, at + 2);
  if (pair === "&&" || pair === "||") return 2;
  return text[at] === "|" ? 1 : 0;
}

/** Split on separators outside quotes and not escaped by a backslash. */
function splitUnquoted(text: string, separatorAt: Separator): string[] {
  const parts: string[] = [];
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
      parts.push(current);
      current = "";
      i += width - 1;
    } else current += char;
  }
  parts.push(current);
  return parts;
}

/** An `&` in `2>&1`, `>&2` or `&>log` redirects output; it ends no command. */
function isRedirect(text: string, at: number): boolean {
  return /[<>]/.test(text[at - 1] ?? "") || text[at + 1] === ">";
}
