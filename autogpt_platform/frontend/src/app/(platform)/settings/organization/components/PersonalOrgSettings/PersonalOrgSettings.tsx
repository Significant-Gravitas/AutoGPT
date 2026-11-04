"use client";

import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";

import { ConvertOrgSection } from "../ConvertOrgSection/ConvertOrgSection";
import { CreateTeamOrgPrompt } from "../CreateTeamOrgPrompt/CreateTeamOrgPrompt";
import { OrgProfileSection } from "../OrgProfileSection/OrgProfileSection";
import { usePersonalOrgSettings } from "./usePersonalOrgSettings";

interface Props {
  org: OrgResponse;
  isAdmin: boolean;
  isOwner: boolean;
  onOrgChanged: () => void;
}

// A personal organization has one member, so it has no members, invitations
// or billing tabs: only its profile and the ways into a team organization.
export function PersonalOrgSettings({
  org,
  isAdmin,
  isOwner,
  onOrgChanged,
}: Props) {
  const { canCreateOrganization } = usePersonalOrgSettings();

  return (
    <div className="flex flex-col gap-8">
      <OrgProfileSection org={org} isAdmin={isAdmin} onSaved={onOrgChanged} />
      {canCreateOrganization ? <CreateTeamOrgPrompt /> : null}
      {canCreateOrganization && isOwner ? (
        <ConvertOrgSection org={org} />
      ) : null}
    </div>
  );
}
