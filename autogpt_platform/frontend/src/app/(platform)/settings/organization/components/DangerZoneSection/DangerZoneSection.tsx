"use client";

import { useState } from "react";

import { useDeleteV2DeleteOrganization } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { toast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useOrgTeamStore } from "@/services/org-team/store";

interface Props {
  org: OrgResponse;
  currentMember: OrgMemberResponse | null;
}

export function DangerZoneSection({ org, currentMember }: Props) {
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const { orgs, setOrgs, setActiveOrg } = useOrgTeamStore();

  const { mutateAsync: deleteOrg, isPending } = useDeleteV2DeleteOrganization({
    mutation: {
      onError: (error) => {
        toast({
          title: "Failed to delete organization",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  if (!currentMember?.is_owner) {
    return null;
  }

  async function handleDeleteConfirmed() {
    await deleteOrg({ orgId: org.id });
    const remaining = orgs.filter((o) => o.id !== org.id);
    setOrgs(remaining);
    const personal = remaining.find((o) => o.isPersonal) ?? remaining[0];
    if (personal) {
      setActiveOrg(personal.id);
    }
    getQueryClient().resetQueries();
    toast({ title: `Organization "${org.name}" deleted`, variant: "success" });
    setIsConfirmOpen(false);
  }

  return (
    <section
      className="flex flex-col gap-3 rounded-xl border border-red-200 p-4"
      data-testid="org-danger-zone"
    >
      <Text variant="h4" as="h2">
        Danger zone
      </Text>
      <div className="flex items-center gap-3">
        <div className="flex flex-1 flex-col">
          <Text variant="body">Delete this organization</Text>
          <Text variant="small" className="text-zinc-500">
            Members lose access; financial records are retained.
          </Text>
        </div>
        <Button variant="destructive" onClick={() => setIsConfirmOpen(true)}>
          Delete organization
        </Button>
      </div>

      <Dialog
        title={`Delete ${org.name}?`}
        styling={{ maxWidth: "26rem" }}
        controlled={{
          isOpen: isConfirmOpen,
          set: (open) => {
            if (!isPending) setIsConfirmOpen(open);
          },
        }}
      >
        <Dialog.Content>
          <Text variant="body">
            This removes access for all {org.member_count} member
            {org.member_count === 1 ? "" : "s"}. This cannot be undone from the
            UI.
          </Text>
          <div className="flex justify-end gap-2 pt-4">
            <Button
              variant="secondary"
              onClick={() => setIsConfirmOpen(false)}
              disabled={isPending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={isPending}
              onClick={handleDeleteConfirmed}
            >
              Delete organization
            </Button>
          </div>
        </Dialog.Content>
      </Dialog>
    </section>
  );
}
