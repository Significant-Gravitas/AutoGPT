import type { SessionSummary } from "@/lib/api/chat";
import type { Expert } from "@/lib/api/experts";
import { OTTO_AVATAR_URL, OTTO_NAME, OTTO_ROLE } from "@/lib/otto";
import { avatarSource, roleLine } from "@/features/team/helpers";

// The list route's maximum page size.
export const DRAWER_SESSION_LIMIT = 100;

export interface ChatGroup {
  key: string;
  label: string;
  role: string | null;
  avatar: string | null;
  isOtto: boolean;
  sessions: SessionSummary[];
}

// Same order as the web sidebar: Otto first, then each expert in the order
// their most recent chat appears in the list.
export function groupChatsByExpert(
  sessions: SessionSummary[],
  experts: Expert[],
): ChatGroup[] {
  const byExpert = new Map<string | null, SessionSummary[]>();
  for (const session of sessions) {
    const key = session.expert_id ?? null;
    const bucket = byExpert.get(key);
    if (bucket) bucket.push(session);
    else byExpert.set(key, [session]);
  }

  const expertsByID = new Map(experts.map((expert) => [expert.id, expert]));
  const groups = [...byExpert.entries()].map(([expertID, grouped]) =>
    toGroup(expertID, grouped, expertsByID),
  );
  return [
    ...groups.filter((group) => group.isOtto),
    ...groups.filter((group) => !group.isOtto),
  ];
}

// Past the first page the drawer can't know a group's real count.
export function chatCountLabel(count: number, isTruncated: boolean) {
  return isTruncated ? `${count}+` : String(count);
}

function toGroup(
  expertID: string | null,
  sessions: SessionSummary[],
  expertsByID: Map<string, Expert>,
): ChatGroup {
  if (!expertID) {
    return {
      key: "otto",
      label: OTTO_NAME,
      role: OTTO_ROLE,
      avatar: OTTO_AVATAR_URL,
      isOtto: true,
      sessions,
    };
  }
  const expert = expertsByID.get(expertID);
  return {
    key: expertID,
    label: expert?.name || "Expert",
    role: expert ? roleLine(expert) : null,
    avatar: expert ? avatarSource(expert) : null,
    isOtto: false,
    sessions,
  };
}
