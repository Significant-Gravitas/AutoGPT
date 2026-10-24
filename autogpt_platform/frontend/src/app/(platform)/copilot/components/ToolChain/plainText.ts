// Shell commands, matched by the forms they take rather than as bare words,
// so "the cat food listings" or "the head office rota" stay plain text.
const SHELL =
  "cat|head|tail|less|wc|sort|uniq|cut|tr|find|xargs|touch|rm|cp|mv|ln|tar|zip|unzip|gzip|wget|curl|ssh|scp|rsync|python3?|node|pip3?|npm|pnpm|yarn|git|docker|make|chmod|chown|mkdir|sudo|sed|awk|grep|bash|sh|ls|cd|echo|printf|export|jq";

const TECHNICAL_PATTERNS = [
  // Paths: "~/x", "./x", "/x", or any token with two slashes in it.
  /(?:^|\s)[~.]?\/\w/,
  /\S+\/\S*\//,
  // "reports/brief": two lowercase words joined by a slash, unless the pair
  // is ordinary writing ("and/or", "his/her"). "24/7" and "Instagram/Shopify"
  // never match.
  /\b(?!(?:and|or|he|she|his|her|him|them)\/)[a-z][\w-]+\/(?!(?:and|or|he|she|his|her|him|them)\b)[a-z][\w-]+/,
  /\b[\w-]+\.(?:md|csv|tsv|jsonl?|py|sh|txt|ya?ml|toml|html|js|ts|log|xlsx)\b/i,
  /`/,
  /&&|\|\||\s\|\s|\$\(/,
  /\b(?:sed|awk|grep|mkdir|chmod|sudo|curl|npm|pip3?|bash)\b|\bcd\s|\bls\b/,
  // A command with a flag or a path after it: "head -n 5", "wc -l", "cp ./a".
  new RegExp(`\\b(?:${SHELL})\\s+(?:-{1,2}\\w|[~./])`),
  /\bgit\s+(?:status|log|diff|add|commit|push|pull|clone|checkout|branch|show|stash|reset|merge|rebase)\b/,
  /\b(?:npm|pnpm|yarn|pip3?)\s+(?:install|run|ci|add)\b/,
  // A text that opens with a command and its argument: "cat brief".
  new RegExp(`^(?:${SHELL})\\s+\\S`),
  // Tool and argument names: snake_case identifiers.
  /\b[a-z]+_[a-z_]+\b/,
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/i,
];

/** Whether text a business owner would see names a path, file, shell
 *  command, tool name or internal id: detail that belongs behind "Show
 *  technical details", never in a label or a summary. */
export function looksTechnical(text: string): boolean {
  return TECHNICAL_PATTERNS.some((pattern) => pattern.test(text));
}
