"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ToggleChip } from "@/components/atoms/ToggleChip/ToggleChip";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import {
  GlobeIcon,
  MailAtSign01Icon,
  SecurityBlockIcon,
} from "@hugeicons/core-free-icons";
import { AdminSection } from "../../../components/AdminSection";
import { getErrorMessage } from "../../helpers";
import { BlockRow } from "./components/BlockRow";
import { type KindFilter, useBlocksTab } from "./useBlocksTab";

const KIND_CHIPS: Array<{
  value: KindFilter;
  label: string;
  icon: typeof GlobeIcon;
}> = [
  { value: "all", label: "All", icon: SecurityBlockIcon },
  { value: "IP", label: "IP addresses", icon: GlobeIcon },
  { value: "EMAIL_DOMAIN", label: "Domains", icon: MailAtSign01Icon },
];

export function BlocksTab() {
  const tab = useBlocksTab();

  if (tab.isLoading) {
    return (
      <AdminSection>
        <div className="flex flex-col gap-2" aria-busy="true">
          <Skeleton className="h-10 w-full rounded-lg" />
          <Skeleton className="h-10 w-full rounded-lg" />
          <Skeleton className="h-10 w-full rounded-lg" />
        </div>
      </AdminSection>
    );
  }
  if (tab.isError) {
    return (
      <ErrorCard
        context="blocks"
        responseError={{ message: getErrorMessage(tab.error) }}
        onRetry={() => void tab.refetch()}
      />
    );
  }
  if (tab.blocks.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-16 text-center">
        <Icon icon={SecurityBlockIcon} size={32} className="text-zinc-400" />
        <Text variant="large-medium">No active blocks</Text>
        <Text variant="body" className="max-w-[420px] text-zinc-600">
          Block an address from an account&apos;s Sign-ins, or an email domain
          from its Related accounts.
        </Text>
      </div>
    );
  }

  return (
    <AdminSection>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="min-w-[240px] flex-1">
          <SearchInput
            size="small"
            value={tab.search}
            onChange={tab.setSearch}
            placeholder="Filter by address, domain or reason"
            aria-label="Filter blocks"
          />
        </div>
        <div className="flex items-center gap-1 rounded-full bg-zinc-100 p-0.5">
          {KIND_CHIPS.map((chip) => (
            <ToggleChip
              key={chip.value}
              icon={<Icon icon={chip.icon} size={14} />}
              label={chip.label}
              tooltip={`Show ${chip.label.toLowerCase()}`}
              ariaLabel={chip.label}
              pressed={tab.kind === chip.value}
              onToggle={() => tab.setKind(chip.value)}
            />
          ))}
        </div>
      </div>
      {tab.visible.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          No blocks match.
        </Text>
      ) : (
        <ul className="divide-y divide-zinc-200">
          {tab.visible.map((block) => (
            <BlockRow key={block.id} block={block} />
          ))}
        </ul>
      )}
      {tab.hasMore ? (
        <div className="mt-3 flex justify-center">
          <Button variant="ghost" size="small" onClick={tab.showMore}>
            Show 50 more
          </Button>
        </div>
      ) : null}
    </AdminSection>
  );
}
