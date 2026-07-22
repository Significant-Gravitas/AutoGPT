"use client";

import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";

import { HeldMemoryReviewQueue } from "./components/HeldMemoryReviewQueue/HeldMemoryReviewQueue";
import { useSharedMemorySection } from "./useSharedMemorySection";

interface Props {
  org: OrgResponse;
  isAdmin: boolean;
  onSaved: () => void;
}

// Org-admin surface for shared (org-wide) memory governance: the hold-for-review
// toggle persists to the org, and the review queue lists tentative memories for
// approve/reject.
export function SharedMemorySection({ org, isAdmin, onSaved }: Props) {
  const { holdForReview, isPending, handleToggle } = useSharedMemorySection({
    org,
    onSaved,
  });

  if (!isAdmin) {
    return null;
  }

  return (
    <section
      className="flex flex-col gap-4"
      data-testid="org-shared-memory-section"
    >
      <div className="flex flex-col gap-1">
        <Text variant="h4" as="h2">
          Shared memory
        </Text>
        <Text variant="body" className="text-zinc-500">
          When shared memory is on, facts your organization’s agents learn are
          held as <em>tentative</em> until an admin reviews them — so nothing is
          trusted org-wide without a check.
        </Text>
      </div>

      <div className="rounded-large border border-zinc-200 p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex flex-col">
            <Text variant="body-medium">Hold new memories for review</Text>
            <Text variant="small" className="text-zinc-500">
              New memories stay tentative until an admin approves them.
            </Text>
          </div>
          <Switch
            checked={holdForReview}
            disabled={isPending}
            onCheckedChange={handleToggle}
            aria-label="Hold new memories for review"
          />
        </div>
      </div>

      <HeldMemoryReviewQueue orgId={org.id} />
    </section>
  );
}
