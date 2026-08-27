"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { useActiveMemoryList } from "./useActiveMemoryList";

interface Props {
  orgId: string;
}

export function ActiveMemoryList({ orgId }: Props) {
  const {
    items,
    isLoading,
    isError,
    refetch,
    selectedMemoryId,
    setSelectedMemoryId,
    confirmRevoke,
    isRevoking,
  } = useActiveMemoryList(orgId);
  const selectedMemory = items.find((item) => item.id === selectedMemoryId);

  return (
    <>
      <div
        className="rounded-large border border-zinc-200 p-4"
        data-testid="org-active-memory-list"
      >
        <Text variant="body-medium">Active shared memory</Text>
        <Text variant="small" className="mt-1 block text-zinc-500">
          Facts currently available to the organization’s agents and teams.
        </Text>

        <div className="mt-3">
          {isLoading ? (
            <Skeleton className="h-24 w-full" />
          ) : isError ? (
            <ErrorCard
              responseError={{ message: "Failed to load active memories" }}
              context="active shared memory list"
              onRetry={() => refetch()}
            />
          ) : items.length === 0 ? (
            <Text variant="small" className="text-zinc-500">
              No active shared memories yet.
            </Text>
          ) : (
            <ul className="flex flex-col gap-3">
              {items.map((item) => (
                <li
                  key={item.id}
                  className="flex flex-col gap-2 rounded-medium border border-zinc-100 p-3 sm:flex-row sm:items-start sm:justify-between"
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
                        <Text
                          variant="small"
                          as="span"
                          className="text-zinc-500"
                        >
                          Organization
                        </Text>
                      )}
                    </div>
                    <Text variant="small" className="text-zinc-800">
                      {item.fact ?? item.name ?? "Untitled memory"}
                    </Text>
                    {(item.provenance ?? item.source_kind) && (
                      <Text variant="small" className="text-zinc-400">
                        {item.provenance ?? item.source_kind}
                      </Text>
                    )}
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="small"
                    disabled={isRevoking}
                    onClick={() => setSelectedMemoryId(item.id)}
                  >
                    Revoke
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <Dialog
        title="Revoke this shared memory?"
        styling={{ maxWidth: "28rem" }}
        controlled={{
          isOpen: selectedMemoryId !== null,
          set: (open) => {
            if (!open && !isRevoking) setSelectedMemoryId(null);
          },
        }}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <Text variant="body" className="text-zinc-800">
              Agents will stop recalling this fact from shared memory. The
              revocation remains in the audit history.
            </Text>
            {selectedMemory && (
              <Text variant="small" className="rounded-medium bg-zinc-50 p-3">
                {selectedMemory.fact ??
                  selectedMemory.name ??
                  "Untitled memory"}
              </Text>
            )}
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="secondary"
                size="small"
                disabled={isRevoking}
                onClick={() => setSelectedMemoryId(null)}
              >
                Cancel
              </Button>
              <Button
                type="button"
                variant="destructive"
                size="small"
                loading={isRevoking}
                disabled={isRevoking}
                onClick={confirmRevoke}
              >
                Revoke memory
              </Button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
