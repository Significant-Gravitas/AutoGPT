"use client";

import { useState } from "react";
import { notFound, useSearchParams } from "next/navigation";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";

import { DangerZoneSection } from "./components/DangerZoneSection/DangerZoneSection";
import { InvitationsSection } from "./components/InvitationsSection/InvitationsSection";
import { MembersSection } from "./components/MembersSection/MembersSection";
import { MyInvitationsSection } from "./components/MyInvitationsSection/MyInvitationsSection";
import { OrgProfileSection } from "./components/OrgProfileSection/OrgProfileSection";
import { SharedMemorySection } from "./components/SharedMemorySection/SharedMemorySection";
import { TeamSpendSection } from "./components/TeamSpendSection/TeamSpendSection";
import { TeamsSection } from "./components/TeamsSection/TeamsSection";
import { useOrganizationSettingsPage } from "./useOrganizationSettingsPage";

// The "General" tab keeps the internal `profile` id so existing ?tab=profile
// deep links stay valid; only the visible label changed.
type OrgSettingsTab =
  | "profile"
  | "members"
  | "invitations"
  | "teams"
  | "billing";

function isOrgSettingsTab(value: string): value is OrgSettingsTab {
  return (
    value === "profile" ||
    value === "members" ||
    value === "invitations" ||
    value === "teams" ||
    value === "billing"
  );
}

function resolveInitialTab(searchParams: URLSearchParams): OrgSettingsTab {
  const tabParam = searchParams.get("tab");
  if (tabParam && isOrgSettingsTab(tabParam)) return tabParam;
  return "profile";
}

export default function OrganizationSettingsPage() {
  const { enabled, ready } = useFlagStatus(Flag.SHOW_ORG_SETTINGS);

  if (!ready) return <OrganizationSettingsSkeleton />;
  if (!enabled) notFound();

  return <OrganizationSettingsPageContent />;
}

function OrganizationSettingsSkeleton() {
  return (
    <div className="flex flex-col gap-6 py-6">
      <Skeleton className="h-8 w-64" />
      <Skeleton className="h-40 w-full" />
      <Skeleton className="h-64 w-full" />
    </div>
  );
}

function OrganizationSettingsPageContent() {
  const searchParams = useSearchParams();
  const {
    org,
    members,
    currentMember,
    isAdmin,
    canManageBilling,
    isLoading,
    isError,
    refetchMembers,
    refetchOrg,
  } = useOrganizationSettingsPage();
  const [activeTab, setActiveTab] = useState<OrgSettingsTab>(() =>
    resolveInitialTab(searchParams),
  );

  function handleTabChange(value: string) {
    if (isOrgSettingsTab(value)) setActiveTab(value);
  }

  if (isLoading) {
    return <OrganizationSettingsSkeleton />;
  }

  if (isError || !org) {
    return (
      <div className="py-6">
        <ErrorCard
          responseError={{ message: "Failed to load organization" }}
          context="organization settings"
          onRetry={() => {
            refetchOrg();
            refetchMembers();
          }}
        />
      </div>
    );
  }

  const header = (
    <div>
      <Text variant="h3" as="h1">
        Organization
      </Text>
      <Text variant="body" className="text-zinc-500">
        Manage {org.name}
        {org.is_personal ? " — your personal organization" : ""}
      </Text>
    </div>
  );

  // Invitations is admin-only and Billing is billing-manager/owner-only, so a
  // member without that right never sees the tab. Fall back to General if they
  // land on a gated tab via a ?tab= deep link.
  const effectiveTab =
    (activeTab === "invitations" && !isAdmin) ||
    (activeTab === "billing" && !canManageBilling)
      ? "profile"
      : activeTab;

  return (
    <div className="flex flex-col gap-8 py-6">
      {header}

      <MyInvitationsSection />

      <TabsLine value={effectiveTab} onValueChange={handleTabChange}>
        <TabsLineList>
          <TabsLineTrigger value="profile">General</TabsLineTrigger>
          <TabsLineTrigger value="members">Members</TabsLineTrigger>
          {isAdmin ? (
            <TabsLineTrigger value="invitations">Invitations</TabsLineTrigger>
          ) : null}
          <TabsLineTrigger value="teams">Teams</TabsLineTrigger>
          {canManageBilling ? (
            <TabsLineTrigger value="billing">Billing</TabsLineTrigger>
          ) : null}
        </TabsLineList>

        <TabsLineContent value="profile" className="flex flex-col gap-8">
          <OrgProfileSection org={org} isAdmin={isAdmin} onSaved={refetchOrg} />
          <SharedMemorySection
            org={org}
            isAdmin={isAdmin}
            onSaved={refetchOrg}
          />
          <DangerZoneSection
            org={org}
            members={members}
            currentMember={currentMember}
            onTransferred={() => {
              refetchMembers();
              refetchOrg();
            }}
          />
        </TabsLineContent>

        <TabsLineContent value="members" className="flex flex-col gap-8">
          <MembersSection
            orgId={org.id}
            members={members}
            currentMember={currentMember}
            isAdmin={isAdmin}
            onChanged={refetchMembers}
          />
        </TabsLineContent>

        {isAdmin ? (
          <TabsLineContent value="invitations" className="flex flex-col gap-8">
            <InvitationsSection orgId={org.id} isAdmin={isAdmin} />
          </TabsLineContent>
        ) : null}

        <TabsLineContent value="teams">
          <TeamsSection
            orgId={org.id}
            orgMembers={members}
            currentMember={currentMember}
          />
        </TabsLineContent>

        {canManageBilling ? (
          <TabsLineContent value="billing" className="flex flex-col gap-8">
            <TeamSpendSection
              orgId={org.id}
              canManageBilling={canManageBilling}
            />
          </TabsLineContent>
        ) : null}
      </TabsLine>
    </div>
  );
}
