import { Expert } from "@/app/api/__generated__/models/expert";
import { AUTOPILOT_ROLE } from "../../helpers";

export interface ChatTarget {
  expertId: string | null;
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
    name: expert.name,
    role: expert.role,
    avatarUrl: expert.avatar_url,
  };
}
