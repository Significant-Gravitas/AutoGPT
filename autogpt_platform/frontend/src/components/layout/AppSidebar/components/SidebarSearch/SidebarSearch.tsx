"use client";

import { useGlobalSearchStore } from "@/app/(platform)/components/GlobalSearchModal/useGlobalSearchStore";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { MagnifyingGlassIcon } from "@/components/icons/pika/adapter";
import { SidebarShortcutHint } from "../SidebarShortcutHint/SidebarShortcutHint";

export function SidebarSearch() {
  const openSearch = useGlobalSearchStore((state) => state.openSearch);

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        tooltip="Search"
        onClick={openSearch}
        className="h-auto rounded-lg p-2 font-normal group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:!gap-0 group-data-[collapsible=icon]:!p-1.5 hover:!bg-zinc-100 [&>svg]:size-4"
      >
        <MagnifyingGlassIcon className="size-4" />
        <span className="truncate group-data-[collapsible=icon]:hidden">
          Search
        </span>
        <SidebarShortcutHint mac={["⇧", "⌘", "K"]} other={["⇧", "Ctrl", "K"]} />
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}
