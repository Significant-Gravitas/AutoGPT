import type { ModelSettings } from "./types";

export interface RequestBody {
  modelSettings: ModelSettings;
  goal: string;
  task?: string;
  tasks?: string[];
  lastTask?: string;
  result?: string;
  completedTasks?: string[];
}
