import type { ListTasksParams } from "@/app/api/__generated__/models/listTasksParams";
import { ListTasksSort } from "@/app/api/__generated__/models/listTasksSort";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Choice } from "@/app/(platform)/tasks/components/ChoiceMenu/ChoiceMenu";
import { statusOf } from "@/app/(platform)/tasks/helpers";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";

/** The board reads this many tasks at a time. */
export const PAGE_SIZE = 50;
/** The API's longest search. */
export const QUERY_MAX = 200;
export const ANYONE = "anyone";
export const OTTO = "otto";
export const ANY_STATUS = "any";
export const DEFAULT_SORT: ListTasksSort = ListTasksSort.updated;

/** The orders the board can be read in; lanes keep it. */
export const SORT_CHOICES = [
  { value: ListTasksSort.updated, label: "Recently updated" },
  { value: ListTasksSort.created, label: "Newest" },
  { value: ListTasksSort.title, label: "Title A–Z" },
];

export interface TaskFilters {
  q: string;
  owner: string;
  status: string;
  sort: string;
}

function sortOf(value: string) {
  return Object.values(ListTasksSort).find((sort) => sort === value);
}

/** A picked order other than the default: the board keeps the server's
 *  order in every lane. */
export function isSorted(filters: TaskFilters) {
  const sort = sortOf(filters.sort);
  return sort !== undefined && sort !== DEFAULT_SORT;
}

export type FilterChoice = Choice;

/** The order the status filter lists them in, as the board reads. */
const STATUS_CHOICES: TaskStatus[] = [
  TaskStatus.NEEDS_INPUT,
  TaskStatus.TODO,
  TaskStatus.IN_PROGRESS,
  TaskStatus.FAILED,
  TaskStatus.DONE,
  TaskStatus.CANCELLED,
];

export function statusChoices(): FilterChoice[] {
  return [
    { value: ANY_STATUS, label: "Any status" },
    ...STATUS_CHOICES.map((status) => ({
      value: status,
      label: statusOf(status).word,
    })),
  ];
}

export function ownerChoices(experts: Expert[]): FilterChoice[] {
  return [
    { value: ANYONE, label: "Anyone" },
    { value: OTTO, label: AUTOPILOT_NAME },
    ...experts.map((expert) => ({ value: expert.id, label: expert.name })),
  ];
}

function isStatus(value: string): value is TaskStatus {
  return STATUS_CHOICES.some((status) => status === value);
}

export function isFiltered(filters: TaskFilters) {
  return (
    filters.q.trim() !== "" ||
    filters.owner !== ANYONE ||
    filters.status !== ANY_STATUS
  );
}

/** The list request for these filters, one page long; the offset is the
 *  page's. Unset filters are left out, never sent as null. */
export function listParams(filters: TaskFilters): ListTasksParams {
  const q = filters.q.trim().slice(0, QUERY_MAX);
  const sort = sortOf(filters.sort);
  return {
    limit: PAGE_SIZE,
    ...(sort && sort !== DEFAULT_SORT ? { sort } : {}),
    ...(q ? { q } : {}),
    ...(filters.owner === OTTO ? { owned_by_otto: true } : {}),
    ...(filters.owner !== OTTO && filters.owner !== ANYONE
      ? { owner_expert_id: filters.owner }
      : {}),
    ...(isStatus(filters.status) ? { statuses: [filters.status] } : {}),
  };
}
