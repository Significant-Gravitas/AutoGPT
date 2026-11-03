import type { SelectOption } from "@/components/atoms/Select/Select";
import { PRIORITY_LABELS } from "../../data/helpers";
import type {
  CreateTaskInput,
  ExpertOption,
  TaskPriority,
} from "../../data/types";
import { ALL_PRIORITIES } from "../../helpers";

export const ME_VALUE = "user";
export const OTTO_VALUE = "otto";
const EXPERT_PREFIX = "expert:";

export function assigneeOptions(experts: ExpertOption[]): SelectOption[] {
  return [
    { value: ME_VALUE, label: "Me" },
    { value: OTTO_VALUE, label: "Otto" },
    ...experts.map((expert) => ({
      value: `${EXPERT_PREFIX}${expert.id}`,
      label: expert.busy
        ? `${expert.name} · busy on another task`
        : expert.name,
    })),
  ];
}

export const PRIORITY_OPTIONS: SelectOption[] = ALL_PRIORITIES.map((p) => ({
  value: p,
  label: PRIORITY_LABELS[p],
}));

export function isPriority(value: string): value is TaskPriority {
  return (ALL_PRIORITIES as string[]).includes(value);
}

export function toAssignee(value: string): CreateTaskInput["assignee"] {
  if (value === OTTO_VALUE) return { type: "otto", id: null };
  if (value.startsWith(EXPERT_PREFIX)) {
    return { type: "expert", id: value.slice(EXPERT_PREFIX.length) };
  }
  return { type: "user", id: null };
}
