import {
  AUTOPILOT_HEX,
  getExpertTopicHex,
} from "@/components/molecules/ExpertAvatar/colors";
import type { ExpertIdentity } from "../../useExpertMap";

interface Args {
  expertIdentity?: ExpertIdentity | null;
  isResolvingExpertIdentity?: boolean;
}

/**
 * The material colour of whoever is answering in this thread, for the
 * thinking loader: the expert's own, or Otto's lavender in a thread with no
 * expert. Null while the roster is still loading, so the loader stays grey
 * rather than claim the thread is Otto's.
 */
export function getThinkerHex({
  expertIdentity,
  isResolvingExpertIdentity = false,
}: Args) {
  if (isResolvingExpertIdentity) return null;
  if (!expertIdentity) return AUTOPILOT_HEX;
  return getExpertTopicHex({
    avatarUrl: expertIdentity.avatarUrl,
    role: expertIdentity.role,
  });
}
