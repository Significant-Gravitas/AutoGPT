"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { useWorkloadIdentities } from "../hooks/useWorkloadIdentities";
import { WorkloadRow } from "../WorkloadRow/WorkloadRow";

interface Props {
  onCreate: () => void;
}

export function WorkloadList({ onCreate }: Props) {
  const { identities, isLoading, isError, error, refetch, isEmpty } =
    useWorkloadIdentities();

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2" aria-label="Loading workloads">
        <Skeleton className="h-20 w-full rounded-2xl" />
        <Skeleton className="h-20 w-full rounded-2xl" />
      </div>
    );
  }
  if (isError) {
    const message = error instanceof Error ? error.message : undefined;
    return (
      <ErrorCard
        context="trusted workloads"
        responseError={message ? { message } : undefined}
        onRetry={() => {
          refetch();
        }}
      />
    );
  }
  if (isEmpty) {
    return (
      <div className="flex flex-col items-start gap-3 rounded-2xl border border-dashed border-zinc-300 p-5">
        <Text variant="body-medium" className="text-textBlack">
          No trusted workloads yet
        </Text>
        <Text variant="small" className="text-zinc-600">
          Add one to call the API from GitHub Actions, Vercel, Google Cloud or
          your own site without an API key.
        </Text>
        <Button variant="secondary" size="small" onClick={onCreate}>
          Add workload
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col divide-y divide-zinc-200 overflow-hidden rounded-2xl border border-zinc-200 bg-white">
      {identities.map((identity) => (
        <WorkloadRow key={identity.id} identity={identity} />
      ))}
    </div>
  );
}
