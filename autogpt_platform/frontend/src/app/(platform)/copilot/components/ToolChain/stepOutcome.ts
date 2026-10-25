// Exit 1 from these means "no" (no match, not there, files differ), not that
// the step broke. Any other non-zero exit, or one from any other command, is a
// failure: `curl -f` exits 22, and scripts report errors on stdout.
const ANSWERS_NO = /^(?:grep|egrep|fgrep|rg|test|\[\[?|diff|cmp)(?:\s|$)/;

/** Whether a shell step failed. Only an expected "no" is let through: exit 1
 *  from a predicate (`grep -q`, `test -f`, `diff`) that runs last and
 *  unconditionally. After `&&` or `||` any link of the chain may have set the
 *  exit code (`cd missing && grep -q …`), so that is a failure. */
export function stepFailed(
  command: string | null,
  output: Record<string, unknown> | null,
): boolean {
  if (!output) return false;
  if (output.type === "error" || output.timed_out === true) return true;
  const exitCode = output.exit_code;
  if (typeof exitCode !== "number" || exitCode === 0) return false;
  if (exitCode !== 1 || !command) return true;
  const last = lastCommand(command);
  return last.chained || !ANSWERS_NO.test(last.text);
}

interface Segment {
  text: string;
  // Reached through `&&` or `||`, so its exit code may be an earlier link's.
  chained: boolean;
}

/** The last command of a list or pipeline, and whether `&&` or `||` led to
 *  it. A separator inside quotes (`grep -Eq "TBD|TODO"`), after a backslash or
 *  in a redirection (`2>&1`) is part of the command. */
function lastCommand(command: string): Segment {
  const segments: Segment[] = [];
  let current = "";
  let chained = false;
  let quote: "'" | '"' | null = null;
  function cut(nextChained: boolean) {
    segments.push({ text: current, chained });
    current = "";
    chained = nextChained;
  }
  for (let i = 0; i < command.length; i++) {
    const char = command[i];
    if (char === "\\" && quote !== "'") {
      current += char + (command[i + 1] ?? "");
      i++;
    } else if (quote) {
      if (char === quote) quote = null;
      current += char;
    } else if (char === "'" || char === '"') {
      quote = char;
      current += char;
    } else if ((char === "&" || char === "|") && command[i + 1] === char) {
      cut(true);
      i++;
    } else if (
      /[|;\n]/.test(char) ||
      (char === "&" && !isRedirect(command, i))
    ) {
      cut(false);
    } else current += char;
  }
  cut(false);
  const last = segments.filter((part) => part.text.trim()).pop();
  return {
    text: (last?.text ?? "").trim().replace(/^!\s*/, ""),
    chained: last?.chained ?? false,
  };
}

/** An `&` in `2>&1`, `>&2` or `&>log` redirects output; it ends no command. */
function isRedirect(command: string, at: number): boolean {
  return /[<>]/.test(command[at - 1] ?? "") || command[at + 1] === ">";
}
