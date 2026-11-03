import type { ExpertOption } from "../../../../data/types";

export interface MentionQuery {
  start: number;
  end: number;
  query: string;
}

export function findMentionQuery(
  value: string,
  caret: number,
): MentionQuery | null {
  const before = value.slice(0, caret);
  const match = /(^|\s)@([^\s@]*)$/.exec(before);
  if (!match) return null;
  const query = match[2];
  return { start: caret - query.length - 1, end: caret, query };
}

export function filterExperts(experts: ExpertOption[], query: string) {
  const needle = query.toLowerCase();
  return experts
    .filter((expert) => expert.name.toLowerCase().startsWith(needle))
    .slice(0, 6);
}

export function insertMention(
  value: string,
  mention: MentionQuery,
  name: string,
) {
  const inserted = `@${name} `;
  return {
    value: value.slice(0, mention.start) + inserted + value.slice(mention.end),
    caret: mention.start + inserted.length,
  };
}
