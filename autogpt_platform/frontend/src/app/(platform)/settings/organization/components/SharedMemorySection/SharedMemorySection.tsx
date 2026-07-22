"use client";

import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  isAdmin: boolean;
}

// Org-admin surface for shared (org-wide) memory governance. The hold-for-review
// setting and the tentative-memory review queue both depend on backend that
// doesn't exist yet (memory tiers shipped per-user admin tools only, and the
// org update endpoint can't persist a settings JSON), so the control is shown
// disabled and the review queue is flagged as blocked rather than faked.
export function SharedMemorySection({ isAdmin }: Props) {
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
            checked={false}
            disabled
            aria-label="Hold new memories for review"
          />
        </div>
        <Text variant="small" className="mt-3 block text-amber-600">
          Not yet available — turning this on requires backend support (see
          below).
        </Text>
      </div>

      <div
        className="rounded-large border border-dashed border-zinc-300 bg-zinc-50 p-4"
        data-testid="org-memory-review-blocked"
      >
        <Text variant="body-medium">Review queue</Text>
        <Text variant="small" className="mt-1 block text-zinc-500">
          The admin review queue for tentative memories isn’t available yet.
          It’s blocked on backend endpoints that don’t exist — the memory-tiers
          backend shipped per-user admin tools (<code>/api/admin/memory/*</code>
          ) only and deferred org review. Still needed:
        </Text>
        <ul className="mt-2 list-inside list-disc">
          <li>
            <Text variant="small" as="span" className="text-zinc-500">
              persist the hold-for-review setting on the org (no{" "}
              <code>settings</code> field on the org update endpoint today)
            </Text>
          </li>
          <li>
            <Text variant="small" as="span" className="text-zinc-500">
              list tentative memories for an org/team
            </Text>
          </li>
          <li>
            <Text variant="small" as="span" className="text-zinc-500">
              approve / reject a tentative memory
            </Text>
          </li>
        </ul>
      </div>
    </section>
  );
}
