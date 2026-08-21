"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { useHeldMemoryReviewQueue } from "./useHeldMemoryReviewQueue";

interface Props {
  orgId: string;
}

// Admin review queue for tentative ("held") shared memories. The parent section
// is already admin-gated, so this renders only where a reviewer can act.
export function HeldMemoryReviewQueue({ orgId }: Props) {
  const {
    items,
    isLoading,
    isError,
    refetch,
    handleApprove,
    handleReject,
    isMutating,
  } = useHeldMemoryReviewQueue(orgId);

  return (
    <div
      className="rounded-large border border-zinc-200 p-4"
      data-testid="org-memory-review-queue"
    >
      <Text variant="body-medium">Review queue</Text>
      <Text variant="small" className="mt-1 block text-zinc-500">
        Tentative memories awaiting an admin decision before they’re trusted
        org-wide.
      </Text>

      <div className="mt-3">
        {isLoading ? (
          <Skeleton className="h-24 w-full" />
        ) : isError ? (
          <ErrorCard
            responseError={{ message: "Failed to load review queue" }}
            context="held memory review queue"
            onRetry={() => refetch()}
          />
        ) : items.length === 0 ? (
          <Text
            variant="small"
            className="text-zinc-500"
            data-testid="org-memory-review-empty"
          >
            Nothing awaiting review.
          </Text>
        ) : (
          <ul className="flex flex-col gap-3">
            {items.map((item) => (
              <li
                key={item.id}
                className="flex flex-col gap-2 rounded-medium border border-zinc-100 p-3 sm:flex-row sm:items-start sm:justify-between"
                data-testid="org-memory-review-row"
              >
                <div className="flex flex-col gap-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="info" size="small">
                      {item.tier}
                    </Badge>
                    {item.team_id ? (
                      <Badge variant="info" size="small">
                        {item.team_name ?? "Team"}
                      </Badge>
                    ) : (
                      <Text variant="small" as="span" className="text-zinc-500">
                        Organization
                      </Text>
                    )}
                  </div>
                  <Text variant="small" className="text-zinc-800">
                    {item.fact ?? item.name ?? "Untitled memory"}
                  </Text>
                  {(item.provenance ?? item.source_kind) ? (
                    <Text variant="small" className="text-zinc-400">
                      {item.provenance ?? item.source_kind}
                    </Text>
                  ) : null}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    size="small"
                    disabled={isMutating}
                    onClick={() => handleApprove(item.id)}
                  >
                    Approve
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="small"
                    disabled={isMutating}
                    onClick={() => handleReject(item.id)}
                  >
                    Reject
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
