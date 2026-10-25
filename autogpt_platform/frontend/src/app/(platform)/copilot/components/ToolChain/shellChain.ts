/** The chain that set a shell command's exit status: the list item that
 *  ran last (after any unconditional `;`, newline or `&`), split into its
 *  `&&`/`||` links. Each link is a pipeline, whose status is its last
 *  command's, inverted by a leading `!`. */
export interface Link {
  // The pipeline's last command, without any leading `!`.
  command: string;
  negated: boolean;
  // The operator before this link; empty for the first.
  after: "&&" | "||" | "";
}

export function lastChain(command: string): Link[] {
  const items = splitUnquoted(command, isListSeparator).filter((item) =>
    item.text.trim(),
  );
  const last = items[items.length - 1]?.text ?? "";
  const links = splitUnquoted(last, isChainSeparator);
  return links.map((link, index) => {
    const stages = splitUnquoted(link.text, isPipe);
    const tail = stages[stages.length - 1].text.trim();
    const first = stages[0].text.trim();
    const before = index === 0 ? "" : links[index - 1].separator;
    return {
      command: stages.length === 1 ? tail.replace(/^!\s*/, "") : tail,
      negated: /^!\s/.test(first),
      after: before === "&&" || before === "||" ? before : "",
    };
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
