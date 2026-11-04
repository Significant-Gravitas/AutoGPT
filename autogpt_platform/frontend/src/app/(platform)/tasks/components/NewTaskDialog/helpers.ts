import type { CreateTaskInput, ExpertOption } from "../../data/types";

export const OTTO_VALUE = "otto";
const EXPERT_PREFIX = "expert:";
const TITLE_LIMIT = 80;

export interface AssigneeChoice {
  value: string;
  label: string;
  avatarUrl: string | null;
  busy: boolean;
}

export function assigneeChoices(experts: ExpertOption[]): AssigneeChoice[] {
  return [
    { value: OTTO_VALUE, label: "Otto", avatarUrl: null, busy: false },
    ...experts.map((expert) => ({
      value: `${EXPERT_PREFIX}${expert.id}`,
      label: expert.name,
      avatarUrl: expert.avatar_url ?? null,
      busy: expert.busy,
    })),
  ];
}

export function resolveChoice(
  choices: AssigneeChoice[],
  value: string | null | undefined,
): AssigneeChoice {
  return (
    choices.find((choice) => choice.value === value) ??
    choices.find((choice) => choice.value === OTTO_VALUE)!
  );
}

export function toAssignee(value: string): CreateTaskInput["assignee"] {
  if (value.startsWith(EXPERT_PREFIX)) {
    return { type: "expert", id: value.slice(EXPERT_PREFIX.length) };
  }
  return { type: "otto", id: null };
}

export function splitTaskText(text: string) {
  const full = text.trim();
  const firstLine = full.split("\n")[0].trim();
  const title = shorten(firstLine);
  return { title, description: title === full ? undefined : full };
}

function shorten(line: string) {
  if (line.length <= TITLE_LIMIT) return line;
  const cut = line.slice(0, TITLE_LIMIT);
  const lastSpace = cut.lastIndexOf(" ");
  return `${(lastSpace > TITLE_LIMIT / 2 ? cut.slice(0, lastSpace) : cut).trimEnd()}…`;
}

export function createdMessage(key: string, choice: AssigneeChoice) {
  return `${key} created · ${choice.label} is on it`;
}
