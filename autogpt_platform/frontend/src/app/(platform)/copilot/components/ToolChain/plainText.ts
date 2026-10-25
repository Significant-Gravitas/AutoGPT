import rules from "./technicalTerms.json";

// The rules live in technicalTerms.json, shared with the expert style eval
// (backend copilot/eval/style/plain_language.py), which renders and applies
// them the same way: what the chat hides is what the eval fails a reply for.
// Both test suites run the file's examples.

type Tier = "certain" | "possible";

const WORDS: Record<string, string[]> = rules.words;
const FRAGMENTS: Record<string, string> = rules.fragments;
const PLACEHOLDER = /\{([A-Z_]+)(?::(exact|noabbr))?\}/g;
const ABBREVIATION_LETTERS = 3;

const SOURCES: Record<Tier, string[]> = {
  certain: rules.certain.map((rule) => render(rule.pattern)),
  possible: rules.possible.map((rule) => render(rule.pattern)),
};
const TESTS = [...SOURCES.certain, ...SOURCES.possible].map(
  (source) => new RegExp(source, "m"),
);

/** Whether text a business owner would see names a path, file, shell
 *  command, tool name or internal id, in any case: detail that belongs
 *  behind "Show technical details", never in a label or a summary. A label
 *  costs nothing to replace, so the possible rules count too. */
export function looksTechnical(text: string): boolean {
  return TESTS.some((pattern) => pattern.test(text));
}

/** The terms each tier finds, in the order the rules run, once each: what
 *  the eval reports for the same text. */
export function findTechnicalTerms(text: string): Record<Tier, string[]> {
  return {
    certain: find(SOURCES.certain, text),
    possible: find(SOURCES.possible, text),
  };
}

function find(sources: string[], text: string): string[] {
  const found: string[] = [];
  let remaining = text;
  for (const source of sources) {
    const pattern = new RegExp(source, "gm");
    for (const match of remaining.matchAll(pattern))
      found.push(match.slice(1).find(Boolean) ?? match[0]);
    remaining = remaining.replace(pattern, " ");
  }
  return [...new Set(found)];
}

function render(pattern: string): string {
  const expanded = Object.entries(FRAGMENTS).reduce(
    (result, [name, fragment]) => result.replaceAll(`{${name}}`, fragment),
    pattern,
  );
  return expanded.replace(PLACEHOLDER, (_, name: string, mode?: string) =>
    [...WORDS[name]]
      .sort((a, b) => b.length - a.length)
      .map((word) => wordPattern(word, mode ?? "any"))
      .join("|"),
  );
}

/** A word as written, or in any case. A short word in capitals ("CD", "RM")
 *  reads as an abbreviation, so `noabbr` matches it only in lower or title
 *  case. */
function wordPattern(word: string, mode: string): string {
  if (mode === "exact") return escape(word);
  if (
    mode === "noabbr" &&
    word.replace(/[^A-Za-z]/g, "").length <= ABBREVIATION_LETTERS
  ) {
    const lower = word.toLowerCase();
    return `(?:${escape(lower)}|${escape(lower.charAt(0).toUpperCase() + lower.slice(1))})`;
  }
  return [...word]
    .map((char) =>
      /[A-Za-z]/.test(char)
        ? `[${char.toLowerCase()}${char.toUpperCase()}]`
        : escape(char),
    )
    .join("");
}

function escape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
}
