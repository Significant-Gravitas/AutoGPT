import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { PageField } from "@/app/api/__generated__/models/pageField";

const ACTIVE_STATUSES: ReadonlySet<string> = new Set([
  AgentExecutionStatus.QUEUED,
  AgentExecutionStatus.RUNNING,
  AgentExecutionStatus.INCOMPLETE,
  AgentExecutionStatus.REVIEW,
]);

export function isRunActive(status: string | null | undefined) {
  return Boolean(status && ACTIVE_STATUSES.has(status));
}

export interface FieldGroup {
  name: string | null;
  fields: PageField[];
}

/** Consecutive fields that share a group show under one heading. */
export function groupFields(fields: PageField[]): FieldGroup[] {
  const groups: FieldGroup[] = [];
  for (const field of fields) {
    const name = field.group ?? null;
    const last = groups[groups.length - 1];
    if (last && last.name === name) last.fields.push(field);
    else groups.push({ name, fields: [field] });
  }
  return groups;
}

export function defaultValues(fields: PageField[]) {
  return Object.fromEntries(
    fields
      .filter((field) => field.default !== null && field.default !== undefined)
      .map((field) => [field.input, field.default]),
  );
}
