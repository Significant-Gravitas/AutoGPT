import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import {
  getHiredExperts,
  isSettingUp,
  SETUP_POLL_MS,
} from "@/app/(platform)/team/helpers";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { getIntroTeamLead, getMemberStatus } from "./helpers";

export function useIntroTeam() {
  const { data: experts = [] } = useListExperts({
    query: {
      select: (res) => getHiredExperts((okData(res) ?? []) as Expert[]),
      refetchInterval: (query) =>
        isSettingUp((okData(query.state.data) ?? []) as Expert[])
          ? SETUP_POLL_MS
          : false,
    },
  });

  function trackChatOpened(expert: Expert) {
    trackFunnel("hired_expert_chat_opened", {
      expert_id: expert.id,
      surface: "intro",
      setup_status: getMemberStatus(expert),
    });
  }

  return {
    experts,
    lead: getIntroTeamLead(experts),
    trackChatOpened,
  };
}
