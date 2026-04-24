"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  deleteV1DeleteCredentials,
  getGetV1ListCredentialsQueryKey,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { toast } from "@/components/molecules/Toast/use-toast";

export interface DeleteIntegrationTarget {
  id: string;
  provider: string;
}

export function useDeleteIntegration() {
  const queryClient = useQueryClient();
  const [isPending, setIsPending] = useState(false);

  async function remove(targets: DeleteIntegrationTarget[], force = false) {
    if (targets.length === 0) return;

    setIsPending(true);
    try {
      const results = await Promise.allSettled(
        targets.map((t) =>
          deleteV1DeleteCredentials(t.provider, t.id, force ? { force } : undefined),
        ),
      );

      const failures: string[] = [];
      const needsConfirmation: string[] = [];

      results.forEach((r, idx) => {
        if (r.status === "rejected") {
          failures.push(targets[idx].id);
          return;
        }
        const body = r.value.status === 200 ? r.value.data : null;
        if (body && "need_confirmation" in body && body.need_confirmation) {
          needsConfirmation.push(body.message);
        }
      });

      const successCount = targets.length - failures.length - needsConfirmation.length;

      if (successCount > 0) {
        toast({
          title:
            successCount === 1
              ? "Integration removed"
              : `${successCount} integrations removed`,
          variant: "success",
        });
      }

      if (needsConfirmation.length > 0) {
        toast({
          title: "Confirmation required",
          description: needsConfirmation[0],
          variant: "destructive",
        });
      }

      if (failures.length > 0) {
        toast({
          title: "Some integrations could not be removed",
          description: `${failures.length} of ${targets.length} failed.`,
          variant: "destructive",
        });
      }

      await queryClient.invalidateQueries({
        queryKey: getGetV1ListCredentialsQueryKey(),
      });
    } finally {
      setIsPending(false);
    }
  }

  return { remove, isPending };
}
