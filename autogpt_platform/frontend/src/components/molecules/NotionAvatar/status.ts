export type AvatarStatus =
  | "idle"
  | "thinking"
  | "working"
  | "waiting"
  | "done"
  | "failed"
  | "sleeping";

export interface StatusOption {
  id: AvatarStatus;
  label: string;
  hint: string;
}

export const STATUSES: StatusOption[] = [
  { id: "idle", label: "Idle", hint: "nothing in flight" },
  { id: "thinking", label: "Thinking", hint: "working out what to do" },
  { id: "working", label: "Working", hint: "a run is in flight" },
  { id: "waiting", label: "Needs you", hint: "blocked, asking" },
  { id: "done", label: "Done", hint: "finished its last job" },
  { id: "failed", label: "Failed", hint: "the last run broke" },
  { id: "sleeping", label: "Paused", hint: "schedules off" },
];

export function isAvatarStatus(value: string): value is AvatarStatus {
  return STATUSES.some((status) => status.id === value);
}
