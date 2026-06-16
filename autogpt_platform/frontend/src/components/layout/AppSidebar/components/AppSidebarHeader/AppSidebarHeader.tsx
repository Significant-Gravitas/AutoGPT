"use client";

import { GlobalSearchModal } from "@/app/(platform)/components/GlobalSearchModal/GlobalSearchModal";
import type { SearchResultItem } from "@/app/api/__generated__/models/searchResultItem";
import { SidebarHeader } from "@/components/ui/sidebar";
import { GearIcon, MagnifyingGlassIcon } from "@phosphor-icons/react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

export function AppSidebarHeader() {
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
    <SidebarHeader className="flex animate-fade-in flex-row items-center justify-between p-2">
      <Image
        src="/agpt-logo.png"
        alt="AutoGPT"
        width={40}
        height={40}
        className="size-10"
      />
      <div className="flex items-center gap-2 [&_a]:size-8 [&_a]:border [&_a]:border-zinc-200 [&_a]:bg-white [&_a]:p-1.5 [&_button]:size-8 [&_button]:border [&_button]:border-zinc-200 [&_button]:bg-white [&_button]:p-1.5 [&_svg]:!size-4">
        <Link
          href="/settings"
          aria-label="Settings"
          className="flex items-center justify-center rounded-full text-zinc-600 transition-colors hover:bg-zinc-100"
        >
          <GearIcon className="text-black" />
        </Link>
        <button
          type="button"
          aria-label="Search"
          onClick={() => setIsSearchOpen(true)}
          className="flex items-center justify-center rounded-full p-2 transition-colors hover:bg-white"
        >
          <MagnifyingGlassIcon className="text-black" />
        </button>
      </div>

      <GlobalSearchModal
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onSelectItem={handleSelectSearchItem}
      />
    </SidebarHeader>
  );
}
