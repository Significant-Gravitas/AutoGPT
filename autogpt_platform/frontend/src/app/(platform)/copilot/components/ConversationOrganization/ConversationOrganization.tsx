import { useOrgTeamStore } from "@/services/org-team/store";

interface Props {
  organizationId: string | null;
}

export function ConversationOrganization({ organizationId }: Props) {
  const { activeOrgID, orgs } = useOrgTeamStore();
  if (!organizationId || organizationId === activeOrgID) return null;
  const org = orgs.find((item) => item.id === organizationId);
  return (
    <div
      role="status"
      className="shrink-0 border-b border-border-base bg-canvas px-5 py-2 text-body text-foreground-tertiary"
    >
      Viewing {org?.name ?? "another organization"}
    </div>
  );
}
