import type {
  AgentStatus,
  SitrepPriority,
} from "@/app/(platform)/library/types";

export interface PulseChipData {
  id: string;
  agentID: string;
  name: string;
  status: AgentStatus;
  priority: SitrepPriority;
  shortMessage: string;
}
