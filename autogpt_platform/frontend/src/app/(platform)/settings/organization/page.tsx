"use client";

import { notFound } from "next/navigation";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { MyInvitationsSection } from "./components/MyInvitationsSection/MyInvitationsSection";
import { PersonalOrgSettings } from "./components/PersonalOrgSettings/PersonalOrgSettings";
import { TeamOrgSettings } from "./components/TeamOrgSettings/TeamOrgSettings";
import { useOrganizationSettingsPage } from "./useOrganizationSettingsPage";

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
    </div>
  );
}

function OrganizationSettingsPageContent() {
  const {
    org,
    members,
    currentMember,
    isOwner,
    isAdmin,
    canManageBilling,
    isLoading,
    isError,
    refetchOrg,
    refetchMembers,
  } = useOrganizationSettingsPage();
  if (isLoading) return <OrganizationSettingsSkeleton />;
  if (isError || !org)
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
  return (
    <div className="flex flex-col gap-8 py-6">
      <div>
        <Text variant="h3" as="h1">
          Organization
        </Text>
        <Text variant="body" tone="muted">
          Manage {org.name}
          {org.is_personal ? ", your personal organization" : ""}
        </Text>
      </div>
      <MyInvitationsSection />
      {org.is_personal ? (
        <PersonalOrgSettings
          org={org}
          isAdmin={isAdmin}
          isOwner={isOwner}
          onOrgChanged={refetchOrg}
        />
      ) : (
        <TeamOrgSettings
          org={org}
          members={members}
          currentMember={currentMember}
          isOwner={isOwner}
          isAdmin={isAdmin}
          canManageBilling={canManageBilling}
          onOrgChanged={refetchOrg}
          onMembersChanged={refetchMembers}
        />
      )}
    </div>
  );
}
