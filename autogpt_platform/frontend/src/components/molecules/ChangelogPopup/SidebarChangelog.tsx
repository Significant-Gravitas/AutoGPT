"use client";

import {
  SidebarFooter,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { SparkleIcon } from "@phosphor-icons/react";
import { createPortal } from "react-dom";

import { ChangelogModal } from "./components/ChangelogModal";
import { useSidebarChangelog } from "./useSidebarChangelog";

export function SidebarChangelog() {
  const {
    entries,
    hasUnseen,
    isOpen,
    open,
    close,
    selectedEntry,
    selectEntry,
    entryMarkdown,
    isLoadingMarkdown,
  } = useSidebarChangelog();

  return (
    <SidebarFooter className="p-2">
      <SidebarMenu>
        <SidebarMenuItem>
          <SidebarMenuButton
            tooltip="What's New"
            onClick={open}
            disabled={entries.length === 0}
            aria-label="What's New"
            className="h-auto rounded-lg p-2 font-normal hover:!bg-zinc-100 [&>svg]:size-5"
          >
            <SparkleIcon
              className="size-5"
              weight={hasUnseen ? "fill" : "regular"}
            />
            <span className="truncate">What&apos;s New</span>
            {hasUnseen ? (
              <span
                aria-hidden="true"
                data-testid="changelog-unseen-dot"
                className="ml-auto size-2 shrink-0 rounded-full bg-violet-500 group-data-[collapsible=icon]:hidden"
              />
            ) : null}
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>

      {/* Portal to <body> so the modal escapes the mobile sidebar Sheet's
          stacking context and focus trap. */}
      {isOpen && typeof document !== "undefined"
        ? createPortal(
            <ChangelogModal
              entries={entries}
              selectedEntry={selectedEntry}
              entryMarkdown={entryMarkdown}
              isLoadingMarkdown={isLoadingMarkdown}
              onSelectEntry={selectEntry}
              onClose={close}
            />,
            document.body,
          )
        : null}
    </SidebarFooter>
  );
}
