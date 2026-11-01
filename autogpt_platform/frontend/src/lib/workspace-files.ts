// Agent graphs the builder tools save as `agent.json` are plumbing, not
// something the user made or asked for, so file lists leave them out.
const HIDDEN_FILE_NAMES = new Set(["agent.json"]);

export function isHiddenWorkspaceFile(file: { name: string }): boolean {
  return HIDDEN_FILE_NAMES.has(file.name.toLowerCase());
}
