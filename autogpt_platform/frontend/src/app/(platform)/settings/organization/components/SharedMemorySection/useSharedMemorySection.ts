"use client";

import { useEffect, useState } from "react";

import { usePatchV2UpdateOrganization } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { toast } from "@/components/molecules/Toast/use-toast";

interface Args {
  org: OrgResponse;
  onSaved: () => void;
}

export function useSharedMemorySection({ org, onSaved }: Args) {
  const [holdForReview, setHoldForReview] = useState(
    org.memory_hold_buffer ?? true,
  );

  useEffect(() => {
    setHoldForReview(org.memory_hold_buffer ?? true);
  }, [org.memory_hold_buffer]);

  const { mutateAsync: updateOrg, isPending } = usePatchV2UpdateOrganization({
    mutation: {
      onError: (error) => {
        toast({
          title: "Failed to update shared memory",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  async function handleToggle(next: boolean) {
    const previous = holdForReview;
    setHoldForReview(next);
    try {
      await updateOrg({ orgId: org.id, data: { memory_hold_buffer: next } });
      toast({
        title: next
          ? "Member memories will be held for review"
          : "Member memories will be active immediately",
        variant: "success",
      });
      onSaved();
    } catch {
      setHoldForReview(previous);
    }
  }

  return { holdForReview, isPending, handleToggle };
}
