"use client";

import { useGlobalSearchStore } from "@/app/(platform)/components/GlobalSearchModal/useGlobalSearchStore";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { MagnifyingGlassIcon } from "@phosphor-icons/react";

export function SidebarSearch() {
  const openSearch = useGlobalSearchStore((state) => state.openSearch);

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        tooltip="Search"
        onClick={openSearch}
        className="font-normal group-data-[collapsible=icon]:!p-1.5 hover:!bg-zinc-200 [&>svg]:size-5"
      >
        <MagnifyingGlassIcon className="size-5" />
        <span className="truncate">Search</span>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}
