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
  name?: string;
}

interface RemoveResult {
  succeeded: DeleteIntegrationTarget[];
  failed: DeleteIntegrationTarget[];
  needsConfirmation: { target: DeleteIntegrationTarget; message: string }[];
}

export function useDeleteIntegration() {
  const queryClient = useQueryClient();
  const [isPending, setIsPending] = useState(false);
  const [pendingIds, setPendingIds] = useState<Record<string, true>>({});

  async function remove(
    targets: DeleteIntegrationTarget[],
    force = false,
  ): Promise<RemoveResult> {
    const empty: RemoveResult = {
      succeeded: [],
      failed: [],
      needsConfirmation: [],
    };
    if (targets.length === 0) return empty;

    setIsPending(true);
    setPendingIds(() => {
      const next: Record<string, true> = {};
      for (const t of targets) next[t.id] = true;
      return next;
    });
    try {
      const results = await Promise.allSettled(
        targets.map((t) =>
          deleteV1DeleteCredentials(
            t.provider,
            t.id,
            force ? { force } : undefined,
          ),
        ),
      );

      const out: RemoveResult = {
        succeeded: [],
        failed: [],
        needsConfirmation: [],
      };

      results.forEach((r, idx) => {
        const target = targets[idx];
        if (r.status === "rejected") {
          out.failed.push(target);
          return;
        }
        const body = r.value.data;
        if (body && "need_confirmation" in body && body.need_confirmation) {
          out.needsConfirmation.push({ target, message: body.message });
          return;
        }
        out.succeeded.push(target);
      });

      const successCount = out.succeeded.length;

      if (successCount === 1) {
        const only = out.succeeded[0];
        toast({
          title: only.name ? `Removed ${only.name}` : "Integration removed",
          variant: "success",
        });
      } else if (successCount > 1) {
        const previewNames = out.succeeded
          .map((t) => t.name)
          .filter(Boolean)
          .slice(0, 3)
          .join(", ");
        toast({
          title: `${successCount} integrations removed`,
          description: previewNames || undefined,
          variant: "success",
        });
      }

      // needsConfirmation is surfaced by the caller as a force-delete dialog,
      // so we deliberately don't toast here — the dialog *is* the prompt.

      if (out.failed.length > 0) {
        const failedNames = out.failed
          .map((t) => t.name ?? `${t.provider}/${t.id.slice(0, 6)}`)
          .slice(0, 3)
          .join(", ");
        const more =
          out.failed.length > 3 ? ` +${out.failed.length - 3} more` : "";
        // needsConfirmation items are pending a force-delete prompt, not
        // failures and not successes — exclude them from the denominator
        // so "X of Y could not be removed" doesn't imply the rest succeeded.
        const attemptableCount = targets.length - out.needsConfirmation.length;
        const allFailedTitle =
          attemptableCount === 1
            ? "Failed to remove integration"
            : `Failed to remove ${attemptableCount} integrations`;
        toast({
          title:
            out.failed.length === attemptableCount
              ? allFailedTitle
              : `${out.failed.length} of ${attemptableCount} could not be removed`,
          description: `${failedNames}${more}. Try again to retry the failed ones.`,
          variant: "destructive",
        });
      }

      await queryClient.invalidateQueries({
        queryKey: getGetV1ListCredentialsQueryKey(),
      });

      return out;
    } finally {
      setIsPending(false);
      setPendingIds({});
    }
  }

  function isDeletingId(id: string): boolean {
    return Boolean(pendingIds[id]);
  }

  return { remove, isPending, isDeletingId };
}
