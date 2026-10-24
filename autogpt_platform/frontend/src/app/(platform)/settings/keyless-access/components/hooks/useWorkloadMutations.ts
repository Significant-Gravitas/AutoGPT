"use client";

import { useQueryClient } from "@tanstack/react-query";

import {
  useDeleteV1RevokeAWorkloadIdentity,
  usePatchV1EnableOrDisableAWorkloadIdentity,
  usePostV1TrustAWorkloadIdentity,
} from "@/app/api/__generated__/endpoints/workload-identity/workload-identity";
import type { SetWorkloadIdentityStatusRequestStatus } from "@/app/api/__generated__/models/setWorkloadIdentityStatusRequestStatus";
import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import type { WorkloadIdentitySpec } from "@/app/api/__generated__/models/workloadIdentitySpec";
import { toast } from "@/components/molecules/Toast/use-toast";

import { WORKLOAD_IDENTITIES_QUERY_KEY } from "./useWorkloadIdentities";

export function useWorkloadMutations() {
  const queryClient = useQueryClient();

  function onSuccess(title: string) {
    queryClient.invalidateQueries({ queryKey: WORKLOAD_IDENTITIES_QUERY_KEY });
    toast({ title, variant: "success" });
  }

  function onError(title: string) {
    return (error: unknown) =>
      toast({
        title,
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
  }

  const create = usePostV1TrustAWorkloadIdentity({
    mutation: {
      onSuccess: () => onSuccess("Workload trusted"),
      onError: onError("Could not trust this workload"),
    },
  });
  const setStatus = usePatchV1EnableOrDisableAWorkloadIdentity({
    mutation: {
      onSuccess: () => onSuccess("Workload updated"),
      onError: onError("Could not update this workload"),
    },
  });
  const revoke = useDeleteV1RevokeAWorkloadIdentity({
    mutation: {
      onSuccess: () => onSuccess("Workload revoked"),
      onError: onError("Could not revoke this workload"),
    },
  });

  async function createIdentity(spec: WorkloadIdentitySpec) {
    // The custom Orval mutator throws on non-2xx, so this is the 200 variant.
    const response = await create.mutateAsync({ data: spec });
    return response.data as WorkloadIdentityInfo;
  }

  async function changeStatus(
    identityId: string,
    status: SetWorkloadIdentityStatusRequestStatus,
  ) {
    await setStatus.mutateAsync({ identityId, data: { status } });
  }

  async function revokeIdentity(identityId: string) {
    await revoke.mutateAsync({ identityId });
  }

  return {
    createIdentity,
    changeStatus,
    revokeIdentity,
    isCreating: create.isPending,
    isChangingStatus: setStatus.isPending,
    isRevoking: revoke.isPending,
  };
}
