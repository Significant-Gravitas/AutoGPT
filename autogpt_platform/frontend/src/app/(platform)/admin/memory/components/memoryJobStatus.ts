import type { CommunityRebuildJobStatus } from "@/app/api/__generated__/models/communityRebuildJobStatus";
import type { DreamJobStatus } from "@/app/api/__generated__/models/dreamJobStatus";
import type { NightlyJobStatus } from "@/app/api/__generated__/models/nightlyJobStatus";

export type AnyJobStatus =
  | DreamJobStatus
  | NightlyJobStatus
  | CommunityRebuildJobStatus;
