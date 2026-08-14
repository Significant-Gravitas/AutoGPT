import {
  getListExpertPodsQueryKey,
  getListExpertsQueryKey,
  useAssignExpertPod,
  useCreateExpertPod,
  useListExpertPods,
  useListExperts,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPodWithMembers } from "@/app/api/__generated__/models/expertPodWithMembers";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getExpertSchedules, groupExpertsByPods } from "./helpers";

interface Args {
  enabled: boolean;
}

export function useTeamPage({ enabled }: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [pickerExpertId, setPickerExpertId] = useState<string | null>(null);
  const [soulExpertId, setSoulExpertId] = useState<string | null>(null);
  const [soulDrawerKey, setSoulDrawerKey] = useState(0);
  const [isNewPodOpen, setIsNewPodOpen] = useState(false);

  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[], enabled },
  });
  const podsQuery = useListExpertPods({
    query: {
      select: (res) => (okData(res) ?? []) as ExpertPodWithMembers[],
      enabled,
    },
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], enabled },
  });

  function invalidateExperts() {
    queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() });
    queryClient.invalidateQueries({ queryKey: getListExpertPodsQueryKey() });
  }

  const { mutate: createPodMutate, isPending: isCreatingPod } =
    useCreateExpertPod({
      mutation: {
        onSuccess: () => {
          invalidateExperts();
          setIsNewPodOpen(false);
        },
        onError: () => {
          toast({ title: "Could not create pod", variant: "destructive" });
        },
      },
    });

  const { mutate: assignPodMutate } = useAssignExpertPod({
    mutation: {
      onSuccess: invalidateExperts,
      onError: () => {
        toast({ title: "Could not move expert", variant: "destructive" });
      },
    },
  });

  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );
  const pods = podsQuery.data ?? [];
  const { groups, ungrouped } = groupExpertsByPods(hiredExperts, pods);

  function schedulesForExpert(expert: Expert) {
    return getExpertSchedules(expert, schedulesQuery.data ?? []);
  }

  function installWorkflow(expertId: string) {
    setPickerExpertId(expertId);
  }

  function closeWorkflowPicker() {
    setPickerExpertId(null);
  }

  function refetch() {
    return Promise.all([
      expertsQuery.refetch(),
      podsQuery.refetch(),
      schedulesQuery.refetch(),
    ]);
  }

  function closeSoul() {
    setSoulExpertId(null);
  }

  function openSoul(expertId: string) {
    setSoulExpertId(expertId);
    setSoulDrawerKey((current) => current + 1);
  }

  function createPod(name: string) {
    createPodMutate({ data: { name } });
  }

  function assignPod(expertId: string, podId: string | null) {
    assignPodMutate({ expertId, data: { pod_id: podId } });
  }

  return {
    hiredExperts,
    pods,
    podGroups: groups,
    ungroupedExperts: ungrouped,
    schedulesForExpert,
    isLoading:
      enabled &&
      (expertsQuery.isLoading ||
        podsQuery.isLoading ||
        schedulesQuery.isLoading),
    isError:
      expertsQuery.isError || podsQuery.isError || schedulesQuery.isError,
    refetch,
    installWorkflow,
    pickerExpertId,
    closeWorkflowPicker,
    soulExpert:
      hiredExperts.find((expert) => expert.id === soulExpertId) ?? null,
    soulDrawerKey,
    openSoul,
    closeSoul,
    isNewPodOpen,
    openNewPod: () => setIsNewPodOpen(true),
    closeNewPod: () => setIsNewPodOpen(false),
    createPod,
    isCreatingPod,
    assignPod,
  };
}
