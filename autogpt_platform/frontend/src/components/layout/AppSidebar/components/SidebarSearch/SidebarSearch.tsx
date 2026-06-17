"use client";

import { GlobalSearchModal } from "@/app/(platform)/components/GlobalSearchModal/GlobalSearchModal";
import type { SearchResultItem } from "@/app/api/__generated__/models/searchResultItem";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { MagnifyingGlassIcon } from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { useState } from "react";

export function SidebarSearch() {
  const router = useRouter();
  const [isSearchOpen, setIsSearchOpen] = useState(false);

  function handleSelectSearchItem(item: SearchResultItem) {
    setIsSearchOpen(false);
    if (item.type === "chat_session") {
      router.push(`/copilot?sessionId=${item.id}`);
      return;
    }
    if (item.type === "library_agent") {
      router.push(`/library/agents/${item.id}`);
      return;
    }
    if (item.type === "store_agent") {
      const metadata = (item.metadata ?? {}) as {
        creator?: string;
        slug?: string;
      };
      if (metadata.creator && metadata.slug) {
        router.push(
          `/marketplace/agent/${encodeURIComponent(metadata.creator)}/${encodeURIComponent(metadata.slug)}`,
        );
      }
      return;
    }
    if (item.type === "workspace_file") {
      window.open(
        `/api/proxy/api/workspace/files/${item.id}/download`,
        "_blank",
        "noopener,noreferrer",
      );
    }
  }

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        tooltip="Search"
        onClick={() => setIsSearchOpen(true)}
        className="font-normal group-data-[collapsible=icon]:!p-1.5 hover:!bg-zinc-200 [&>svg]:size-5"
      >
        <MagnifyingGlassIcon className="size-5" />
        <span className="truncate">Search</span>
      </SidebarMenuButton>

      <GlobalSearchModal
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onSelectItem={handleSelectSearchItem}
      />
    </SidebarMenuItem>
  );
}
