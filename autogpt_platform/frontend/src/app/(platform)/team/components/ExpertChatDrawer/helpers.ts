import { Expert } from "@/app/api/__generated__/models/expert";
import { AUTOPILOT_ROLE } from "../../helpers";

export interface ChatTarget {
  expertId: string | null;
  organizationId?: string | null;
  teamId?: string | null;
  name: string;
  role: string;
  avatarUrl: string | null;
}

export const AUTOPILOT_CHAT_TARGET: ChatTarget = {
  expertId: null,
  name: "Autopilot",
  role: AUTOPILOT_ROLE,
  avatarUrl: null,
};

export function expertToChatTarget(expert: Expert): ChatTarget {
  return {
    expertId: expert.id,
    organizationId: expert.organization_id ?? null,
    teamId: expert.team_id ?? null,
    name: expert.name,
    role: expert.role,
    avatarUrl: expert.avatar_url,
  };
}
