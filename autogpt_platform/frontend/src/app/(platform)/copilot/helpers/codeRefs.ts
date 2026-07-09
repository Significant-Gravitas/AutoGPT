import type { CodeRef } from "../store";

/** Short chip label, e.g. `helpers.ts:12-18`. */
export function codeRefLabel(ref: CodeRef): string {
  const name = ref.path.split("/").filter(Boolean).pop() || ref.path;
  const location =
    ref.fromLine === ref.toLine
      ? `:${ref.fromLine}`
      : `:${ref.fromLine}-${ref.toLine}`;
  return `${name}${location}`;
}

/** Markdown code block prepended to the message so the agent sees the context. */
export function formatCodeRef(ref: CodeRef): string {
  const location =
    ref.fromLine === ref.toLine
      ? `line ${ref.fromLine}`
      : `lines ${ref.fromLine}-${ref.toLine}`;
  return `In \`${ref.path}\` (${location}):\n\`\`\`\n${ref.code}\n\`\`\``;
}
