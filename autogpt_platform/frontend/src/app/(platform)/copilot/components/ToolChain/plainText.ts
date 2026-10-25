// Shell commands, matched by the forms they take rather than as bare words,
// so "the cat food listings" or "the head office rota" stay plain text.
// Commands that are also everyday words ("sort", "export", "find") count as a
// command only before an argument, never before a function word: "Running
// sort on items" is prose, "Running sort orders" is a command.
const WORDLIKE =
  "cat|head|tail|less|sort|cut|tr|find|touch|ln|zip|node|make|echo|export";
const CODE_ONLY =
  "wc|uniq|xargs|rm|cp|mv|tar|unzip|gzip|wget|curl|ssh|scp|rsync|python3?|pip3?|npm|pnpm|yarn|git|docker|chmod|chown|mkdir|sudo|sed|awk|grep|bash|sh|ls|cd|printf|jq";
const SHELL = `${WORDLIKE}|${CODE_ONLY}`;
const FUNCTION_WORDS =
  "on|of|for|to|the|a|an|and|or|in|at|by|from|with|into|about|over|through|this|that|these|those|my|your|our|their|its|all|any|some";

// Verbs that put the next word in command position. Prepositions such as
// "with" are left out: "listings with cat food photos" is prose.
const RUN_VERB =
  "(?:[Rr]un(?:s|ning)?|[Rr]an|[Ee]xecut(?:e|es|ed|ing)|[Cc]all(?:s|ed|ing)?)";

const TECHNICAL_PATTERNS = [
  // Paths: "~/x", "./x", "/x", or any token with two slashes in it.
  /(?:^|\s)[~.]?\/\w/,
  /\S+\/\S*\//,
  /\b[\w-]+\.(?:md|csv|tsv|jsonl?|py|sh|txt|ya?ml|toml|html|log|xlsx)\b/i,
  // A script by its file name ("index.js"); "Next.js" or "Node.js" is a
  // product name, not a file.
  /\b[a-z0-9_][\w-]*\.(?:js|ts)\b/,
  /`/,
  /&&|\|\||\s\|\s|\$\(/,
  /\b(?:sed|awk|grep|mkdir|chmod|sudo|curl|npm|pip3?|bash)\b|\bcd\s|\bls\b/,
  // A command with a flag or a path after it: "head -n 5", "wc -l", "cp ./a".
  new RegExp(`\\b(?:${SHELL})\\s+(?:-{1,2}\\w|[~./])`),
  /\bgit\s+(?:status|log|diff|add|commit|push|pull|clone|checkout|branch|show|stash|reset|merge|rebase)\b/,
  /\b(?:npm|pnpm|yarn|pip3?)\s+(?:install|run|ci|add)\b/,
  // A command and its argument in command position: opening the text ("cat
  // brief") or right after a run verb ("Running cat brief", "ran head …").
  new RegExp(
    `(?:^|\\b${RUN_VERB}\\s+)(?:(?:${CODE_ONLY})\\s+\\S|(?:${WORDLIKE})\\s+(?!(?:${FUNCTION_WORDS})\\b)\\S)`,
  ),
  // Tool and argument names: snake_case identifiers.
  /\b[a-z]+_[a-z_]+\b/,
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/i,
];

// Two words joined by a slash that are ordinary writing, not a folder and a
// file ("reports/brief"). "24/7" and "Instagram/Shopify" never look like a
// path in the first place: a path pair is two lowercase words.
const PROSE_PAIRS = new Set([
  "and/or",
  "he/she",
  "his/her",
  "him/her",
  "she/he",
  "they/them",
  "yes/no",
  "pros/cons",
  "input/output",
  "inputs/outputs",
  "read/write",
  "on/off",
  "true/false",
  "before/after",
  "either/or",
  "in/out",
  "up/down",
  "start/end",
  "open/close",
  "buy/sell",
  "online/offline",
]);
const SLASH_PAIR = /\b([a-z][\w-]*)\/([a-z][\w-]*)\b/g;

/** Whether text a business owner would see names a path, file, shell
 *  command, tool name or internal id: detail that belongs behind "Show
 *  technical details", never in a label or a summary. */
export function looksTechnical(text: string): boolean {
  return (
    TECHNICAL_PATTERNS.some((pattern) => pattern.test(text)) ||
    hasPathPair(text)
  );
}

function hasPathPair(text: string): boolean {
  return [...text.matchAll(SLASH_PAIR)].some(
    ([pair]) => !PROSE_PAIRS.has(pair.toLowerCase()),
  );
}
