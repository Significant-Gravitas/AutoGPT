import type { ExpertKickoffMetadata } from "@/app/(platform)/copilot/expertKickoff";
import type { OutgoingMessage } from "@/app/(platform)/copilot/helpers/sendWithAttachments";
import { Expert } from "@/app/api/__generated__/models/expert";
import { AUTOPILOT_ROLE } from "../../helpers";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";

export interface ChatTarget {
  expertId: string | null;
  name: string;
  role: string;
  avatarUrl: string | null;
  color?: string | null;
}

export const AUTOPILOT_CHAT_TARGET: ChatTarget = {
  expertId: null,
  name: AUTOPILOT_NAME,
  role: AUTOPILOT_ROLE,
  avatarUrl: AUTOPILOT_AVATAR_URL,
};

export function expertToChatTarget(expert: Expert): ChatTarget {
  return {
    expertId: expert.id,
    name: expert.name,
    role: expert.role,
    avatarUrl: expert.avatar_url,
    color: expert.color,
  };
}

export function textMessage(
  text: string,
  metadata?: ExpertKickoffMetadata,
): OutgoingMessage {
  return { text, files: [], parts: [], metadata };
}
