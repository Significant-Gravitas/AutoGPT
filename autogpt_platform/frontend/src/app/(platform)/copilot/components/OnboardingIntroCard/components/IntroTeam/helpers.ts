import type { Expert } from "@/app/api/__generated__/models/expert";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";

export type MemberStatus = "installing" | "ready" | "failed";

export function getIntroTeamLead(experts: Expert[]) {
  return experts.some((expert) => getMemberStatus(expert) === "installing")
    ? "Your team is installing their skills and workflows. You can open their chats now, or keep talking to me meanwhile."
    : "Your team is ready. Open a chat to get started.";
}

export function getMemberStatus(expert: Expert): MemberStatus {
  return expert.setup_status ?? "ready";
}

export function getMemberStatusLabel(status: MemberStatus) {
  if (status === "installing") return "Setting up…";
  if (status === "failed") return "Setup didn't finish";
  return "Ready";
}

export function getMemberArea(expert: Expert) {
  return expert.job_title || getExpertRoleLabel(expert.role);
}

// kickoff=1 opens the expert's latest chat, or starts the one where it
// introduces itself. /home, not /copilot: the greeting is already on /home,
// and /copilot only redirects there.
export function getKickoffHref(expertId: string) {
  return `/home?expertId=${encodeURIComponent(expertId)}&kickoff=1`;
}
