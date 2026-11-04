"use client";

import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";

import { DangerZoneSection } from "../DangerZoneSection/DangerZoneSection";
import { InvitationsSection } from "../InvitationsSection/InvitationsSection";
import { MembersSection } from "../MembersSection/MembersSection";
import { OrgProfileSection } from "../OrgProfileSection/OrgProfileSection";
import { SharedMemorySection } from "../SharedMemorySection/SharedMemorySection";
import { TeamSpendSection } from "../TeamSpendSection/TeamSpendSection";
import { useTeamOrgSettings } from "./useTeamOrgSettings";

interface Props {
  org: OrgResponse;
  members: OrgMemberResponse[];
  currentMember: OrgMemberResponse | null;
  isOwner: boolean;
  isAdmin: boolean;
  canManageBilling: boolean;
  onOrgChanged: () => void;
  onMembersChanged: () => void;
}

export function TeamOrgSettings({
  org,
  members,
  currentMember,
  isOwner,
  isAdmin,
  canManageBilling,
  onOrgChanged,
  onMembersChanged,
}: Props) {
  const { activeTab, canSeeInvitations, canSeeBilling, handleTabChange } =
    useTeamOrgSettings({ isAdmin, canManageBilling });

  return (
    <TabsLine value={activeTab} onValueChange={handleTabChange}>
      <TabsLineList>
        <TabsLineTrigger value="profile">General</TabsLineTrigger>
        <TabsLineTrigger value="members">Members</TabsLineTrigger>
        {canSeeInvitations ? (
          <TabsLineTrigger value="invitations">Invitations</TabsLineTrigger>
        ) : null}
        {canSeeBilling ? (
          <TabsLineTrigger value="billing">Billing</TabsLineTrigger>
        ) : null}
      </TabsLineList>

      <TabsLineContent value="profile">
        <div className="flex flex-col gap-8">
          <OrgProfileSection
            org={org}
            isAdmin={isAdmin}
            onSaved={onOrgChanged}
          />
          <SharedMemorySection
            org={org}
            isAdmin={isAdmin}
            onSaved={onOrgChanged}
          />
          <DangerZoneSection
            org={org}
            members={members}
            currentMember={currentMember}
            onTransferred={() => {
              onMembersChanged();
              onOrgChanged();
            }}
          />
        </div>
      </TabsLineContent>

      <TabsLineContent value="members">
        <MembersSection
          orgId={org.id}
          members={members}
          currentMember={currentMember}
          isAdmin={isAdmin}
          isOwner={isOwner}
          onChanged={onMembersChanged}
        />
      </TabsLineContent>

      {canSeeInvitations ? (
        <TabsLineContent value="invitations">
          <InvitationsSection
            orgId={org.id}
            isAdmin={isAdmin}
            isOwner={isOwner}
          />
        </TabsLineContent>
      ) : null}

      {canSeeBilling ? (
        <TabsLineContent value="billing">
          <TeamSpendSection
            orgId={org.id}
            canManageBilling={canManageBilling}
          />
        </TabsLineContent>
      ) : null}
    </TabsLine>
  );
}
