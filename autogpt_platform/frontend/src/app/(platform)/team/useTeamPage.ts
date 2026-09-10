import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import { useState } from "react";
import {
  AUTOPILOT_CHAT_TARGET,
  ChatTarget,
  expertToChatTarget,
} from "./components/ExpertChatDrawer/helpers";
import { getExpertSchedules, getHiredExperts } from "./helpers";

interface Args {
  enabled: boolean;
}

export function useTeamPage({ enabled }: Args) {
  const [pickerExpertId, setPickerExpertId] = useState<string | null>(null);
  const [soulExpertId, setSoulExpertId] = useState<string | null>(null);
  const [soulDrawerKey, setSoulDrawerKey] = useState(0);
  const [chatTarget, setChatTarget] = useState<ChatTarget | null>(null);
  const [chatDrawerKey, setChatDrawerKey] = useState(0);

  const expertsQuery = useListExperts({
    query: { select: (res) => (okData(res) ?? []) as Expert[], enabled },
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], enabled },
  });
  const hiredExperts = getHiredExperts(expertsQuery.data ?? []);
  const schedules = schedulesQuery.data ?? [];

  function schedulesForExpert(expert: Expert) {
    return getExpertSchedules(expert, schedules);
  }

  function installWorkflow(expertId: string) {
    setPickerExpertId(expertId);
  }

  function closeWorkflowPicker() {
    setPickerExpertId(null);
  }

  function refetch() {
    return Promise.all([expertsQuery.refetch(), schedulesQuery.refetch()]);
  }

  function closeSoul() {
    setSoulExpertId(null);
  }

  function openSoul(expertId: string) {
    setChatTarget(null);
    setSoulExpertId(expertId);
    setSoulDrawerKey((current) => current + 1);
  }

  function openChat(expertId: string | null) {
    const expert = hiredExperts.find((candidate) => candidate.id === expertId);
    setSoulExpertId(null);
    setChatTarget(expert ? expertToChatTarget(expert) : AUTOPILOT_CHAT_TARGET);
    setChatDrawerKey((current) => current + 1);
  }

  function closeChat() {
    setChatTarget(null);
  }

  return {
    hiredExperts,
    schedules,
    schedulesForExpert,
    isLoading: enabled && (expertsQuery.isLoading || schedulesQuery.isLoading),
    isError: expertsQuery.isError || schedulesQuery.isError,
    refetch,
    installWorkflow,
    pickerExpertId,
    closeWorkflowPicker,
    soulExpert:
      hiredExperts.find((expert) => expert.id === soulExpertId) ?? null,
    soulDrawerKey,
    openSoul,
    closeSoul,
    chatTarget,
    chatDrawerKey,
    openChat,
    closeChat,
  };
}
