"use client";

import { useState } from "react";

import { SetWorkloadIdentityStatusRequestStatus } from "@/app/api/__generated__/models/setWorkloadIdentityStatusRequestStatus";
import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import { WorkloadIdentityStatus } from "@/app/api/__generated__/models/workloadIdentityStatus";

import { useWorkloadMutations } from "../hooks/useWorkloadMutations";

export function useWorkloadRow(identity: WorkloadIdentityInfo) {
  const [usageOpen, setUsageOpen] = useState(false);
  const [revokeOpen, setRevokeOpen] = useState(false);
  const { changeStatus, isChangingStatus } = useWorkloadMutations();
  const isDisabled = identity.status === WorkloadIdentityStatus.DISABLED;

  async function toggleStatus() {
    const next = isDisabled
      ? SetWorkloadIdentityStatusRequestStatus.ACTIVE
      : SetWorkloadIdentityStatusRequestStatus.DISABLED;
    try {
      await changeStatus(identity.id, next);
    } catch {
      // The mutation's toast explains the failure.
    }
  }

  return {
    usageOpen,
    setUsageOpen,
    revokeOpen,
    setRevokeOpen,
    isDisabled,
    isChangingStatus,
    toggleStatus,
  };
}
