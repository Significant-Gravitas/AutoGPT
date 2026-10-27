/** One `&&`/`||` link of a chain: a pipeline. Its status is its last
 *  command's (under `pipefail`, its rightmost failing command's), inverted
 *  by a leading `!`. */
export interface Link {
  // The pipeline's commands in order, without the leading `!`.
  stages: string[];
  negated: boolean;
  // The operator before this link; empty for the first.
  after: "&&" | "||" | "";
}

/** A list item (between `;`, newlines and background `&`) and whether
 *  `set -e` was on when it ran, so its failure would have ended the run. */
export interface ShellItem {
  links: Link[];
  errexit: boolean;
}

export function shellItems(command: string): ShellItem[] {
  let errexit = false;
  return splitUnquoted(command, isListSeparator)
    .map((item) => item.text.trim())
    .filter(Boolean)
    .map((text) => {
      const item = { links: chainLinks(text), errexit };
      errexit = errexitAfter(text, errexit);
      return item;
    });
}

function chainLinks(text: string): Link[] {
  const links = splitUnquoted(text, isChainSeparator);
  return links.map((link, index) => {
    const stages = splitUnquoted(link.text, isPipe).map((stage) =>
      stage.text.trim(),
    );
    const negated = /^!\s/.test(stages[0]);
    if (negated) stages[0] = stages[0].replace(/^!\s*/, "");
    const before = index === 0 ? "" : links[index - 1].separator;
    return {
      stages,
      negated,
      after: before === "&&" || before === "||" ? before : "",
    };
  });
}

/** Whether `set -e` is on after a list item: `set -e`, `set -euo pipefail`
 *  and `set -o errexit` turn it on; `set +e` and `set +o errexit` off. */
function errexitAfter(text: string, errexit: boolean): boolean {
  if (!/^set\s/.test(text)) return errexit;
  let state = errexit;
  const words = text.split(/\s+/).slice(1);
  words.forEach((word, index) => {
    const named = words[index + 1] === "errexit";
    if (/^-[a-z]*e[a-z]*$/.test(word) || (word === "-o" && named)) state = true;
    if (/^\+[a-z]*e[a-z]*$/.test(word) || (word === "+o" && named))
      state = false;
  });
  return state;
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
