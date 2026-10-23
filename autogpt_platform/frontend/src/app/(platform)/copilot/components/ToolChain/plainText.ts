const TECHNICAL_PATTERNS = [
  // Paths: "~/x", "./x", "/x", or any token with two slashes in it. One slash
  // alone ("24/7") is ordinary writing.
  /(?:^|\s)[~.]?\/\w/,
  /\S+\/\S*\//,
  /\b[\w-]+\.(?:md|csv|tsv|jsonl?|py|sh|txt|ya?ml|toml|html|js|ts|log|xlsx)\b/i,
  /`/,
  /&&|\|\||\s\|\s|\$\(/,
  /\b(?:sed|awk|grep|mkdir|chmod|sudo|curl|npm|pip3?|bash)\b|\bcd\s|\bls\b/,
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
