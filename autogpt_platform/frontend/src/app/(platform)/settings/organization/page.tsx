"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { AliasesSection } from "./components/AliasesSection/AliasesSection";
import { DangerZoneSection } from "./components/DangerZoneSection/DangerZoneSection";
import { InvitationsSection } from "./components/InvitationsSection/InvitationsSection";
import { MembersSection } from "./components/MembersSection/MembersSection";
import { MyInvitationsSection } from "./components/MyInvitationsSection/MyInvitationsSection";
import { OrgProfileSection } from "./components/OrgProfileSection/OrgProfileSection";
import { TeamsSection } from "./components/TeamsSection/TeamsSection";
import { useOrganizationSettingsPage } from "./useOrganizationSettingsPage";

export default function OrganizationSettingsPage() {
  const {
    org,
    members,
    currentMember,
    isAdmin,
    isLoading,
    isError,
    refetchMembers,
    refetchOrg,
  } = useOrganizationSettingsPage();

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6 py-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (isError || !org) {
    return (
      <div className="py-6">
        <ErrorCard
          responseError={{ message: "Failed to load organization" }}
          context="organization settings"
          onRetry={() => refetchOrg()}
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8 py-6">
      <div>
        <Text variant="h3" as="h1">
          Organization
        </Text>
        <Text variant="body" className="text-zinc-500">
          Manage {org.name}
          {org.is_personal ? " — your personal organization" : ""}
        </Text>
      </div>

      <MyInvitationsSection />

      <OrgProfileSection org={org} isAdmin={isAdmin} onSaved={refetchOrg} />

      {!org.is_personal ? (
        <>
          <MembersSection
            orgId={org.id}
            members={members}
            currentMember={currentMember}
            isAdmin={isAdmin}
            onChanged={refetchMembers}
          />
          <TeamsSection
            orgId={org.id}
            orgMembers={members}
            currentMember={currentMember}
          />
          <InvitationsSection orgId={org.id} isAdmin={isAdmin} />
          <AliasesSection orgId={org.id} isAdmin={isAdmin} />
          <DangerZoneSection
            org={org}
            members={members}
            currentMember={currentMember}
            onTransferred={() => {
              refetchMembers();
              refetchOrg();
            }}
          />
        </>
      ) : (
        <Text variant="body" className="text-zinc-500">
          Personal organizations have a single member. Create a team
          organization from the switcher to invite others.
        </Text>
      )}
    </div>
  );
}
