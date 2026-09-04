import {
  getListExpertPodsQueryKey,
  getListExpertsQueryKey,
  listExperts,
  useAssignExpertPod,
  useCreateExpertPod,
  useListExpertPods,
  useListExperts,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getAssignToastTitle,
  getExpertSchedules,
  groupExpertsByPods,
} from "./helpers";

interface Args {
  enabled: boolean;
}

export function useTeamPage({ enabled }: Args) {
  const queryClient = useQueryClient();
  const [pickerExpertId, setPickerExpertId] = useState<string | null>(null);
  const [soulExpertId, setSoulExpertId] = useState<string | null>(null);
  const [soulDrawerKey, setSoulDrawerKey] = useState(0);
  const [isNewPodOpen, setIsNewPodOpen] = useState(false);

  const expertsQuery = useListExperts({
    query: { select: (res) => (okData(res) ?? []) as Expert[], enabled },
  });
  const podsQuery = useListExpertPods({
    query: {
      select: (res) => (okData(res) ?? []) as ExpertPod[],
      enabled,
    },
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], enabled },
  });
  const pods = podsQuery.data ?? [];
  // Built once per render so each card resolves its pod in O(1) instead of
  // scanning the full list.
  const podsById = new Map(pods.map((pod) => [pod.id, pod]));

  function podForExpert(expert: Expert) {
    return expert.pod_id ? podsById.get(expert.pod_id) : undefined;
  }

  function invalidateExperts() {
    return queryClient.invalidateQueries({
      queryKey: getListExpertsQueryKey(),
    });
  }

  function invalidatePods() {
    return queryClient.invalidateQueries({
      queryKey: getListExpertPodsQueryKey(),
    });
  }

  function writeExpertToCache(updated: Expert) {
    const key = getListExpertsQueryKey();
    const cached =
      queryClient.getQueryData<Awaited<ReturnType<typeof listExperts>>>(key);
    if (cached?.status !== 200) return invalidateExperts();
    queryClient.setQueryData(key, {
      ...cached,
      data: cached.data.map((expert) =>
        expert.id === updated.id ? updated : expert,
      ),
    });
  }

  const { mutate: createPodMutate, isPending: isCreatingPod } =
    useCreateExpertPod({
      mutation: {
        onSuccess: () => {
          void invalidatePods();
          setIsNewPodOpen(false);
        },
        onError: (error) => {
          // The dialog stays open (it only closes on success) so the typed
          // name survives; surface the server's reason, e.g. the 409's
          // "already exists" detail.
          toast({
            title: "Could not create pod",
            description: error instanceof Error ? error.message : undefined,
            variant: "destructive",
          });
        },
      },
    });

  const { mutate: assignPodMutate } = useAssignExpertPod({
    mutation: {
      onSuccess: (response, variables) => {
        // The PATCH returns the updated expert, so patch it into the cached
        // roster instead of refetching the heavy list_experts join.
        if (response.status === 200) writeExpertToCache(response.data);
        else void invalidateExperts();

        const podId = variables.data.pod_id;
        const destination = pods.find((pod) => pod.id === podId);
        // A pod we don't know about means the cached pod list is stale.
        if (podId !== null && !destination) void invalidatePods();
        toast({
          title: getAssignToastTitle({
            podId,
            destinationName: destination?.name,
          }),
        });
      },
      onError: (error) => {
        void invalidatePods();
        toast({
          title: "Could not move expert",
          description: error instanceof Error ? error.message : undefined,
          variant: "destructive",
        });
      },
    },
  });

  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );
  const schedules = schedulesQuery.data ?? [];
  const { groups, ungrouped } = groupExpertsByPods(hiredExperts, pods);

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

  function openNewPod() {
    setIsNewPodOpen(true);
  }

  function closeNewPod() {
    setIsNewPodOpen(false);
  }

  return {
    hiredExperts,
    pods,
    podForExpert,
    podGroups: groups,
    ungroupedExperts: ungrouped,
    schedules,
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
    openNewPod,
    closeNewPod,
    createPod,
    isCreatingPod,
    assignPod,
  };
}
