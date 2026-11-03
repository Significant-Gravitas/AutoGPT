"use client";

import { Text } from "@/components/atoms/Text/Text";
import { AdminSection } from "../../../components/AdminSection";
import { AccessGrantList } from "../AccessGrantList/AccessGrantList";
import { EnableAccessForm } from "../EnableAccessForm/EnableAccessForm";
import { useOrgAccessView } from "./useOrgAccessView";

export function OrgAccessView() {
  const view = useOrgAccessView();

  return (
    <div className="flex w-full max-w-[1120px] flex-col gap-6">
      <div className="flex flex-col gap-1">
        <Text variant="h4" as="h1">
          Organization access
        </Text>
        <Text variant="body" tone="secondary">
          Organizations stay hidden for every account until you turn them on
          here. An account&apos;s team organizations pass access on to their
          members and to the people they invite.
        </Text>
      </div>
      <AdminSection
        title="Turn organizations on"
        description="Find the account by the email it signs in with."
      >
        <EnableAccessForm isEnabling={view.isEnabling} onEnable={view.enable} />
      </AdminSection>
      <AdminSection
        title="Accounts with organizations"
        description="Accounts you turned organizations on for, newest first."
      >
        <AccessGrantList
          grants={view.grants}
          total={view.total}
          isLoading={view.isLoading}
          isError={view.isError}
          onRetry={view.retry}
          search={view.search}
          onSearchChange={view.changeSearch}
          page={view.page}
          onPageChange={view.setPage}
          isDisabling={view.isDisabling}
          onDisable={view.disable}
        />
      </AdminSection>
    </div>
  );
}
