"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Button as ShadcnButton } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import {
  FilesIcon,
  MagnifyingGlassIcon,
  PlusIcon,
} from "@phosphor-icons/react";
import { mockSidebarSessions } from "../../script/mockSidebarSessions";
import { ChatSessionBlock } from "@/app/(platform)/copilot/components/ChatSessionBlock/ChatSessionBlock";

export function TourChatSidebar() {
  return (
    <Sidebar
      variant="inset"
      collapsible="icon"
      className="!top-[calc(50px+var(--preview-banner-height,0px))] !h-[calc(100vh-50px-var(--preview-banner-height,0px))] px-0 [&_[data-sidebar=sidebar]]:border-r [&_[data-sidebar=sidebar]]:border-r-[#80808017]"
    >
      <SidebarHeader className="shrink-0 px-4 pb-3 pt-3">
        <div className="flex flex-col gap-3 px-3">
          <div className="flex items-center justify-between">
            <Text variant="h3" size="body-medium">
              Your chats
            </Text>
            <div className="flex items-center">
              <ShadcnButton
                type="button"
                variant="ghost"
                size="icon-sm"
                aria-label="Search chats"
                className="rounded-full text-zinc-600 hover:bg-zinc-100"
              >
                <MagnifyingGlassIcon className="!size-5" />
              </ShadcnButton>
              <ShadcnButton
                type="button"
                variant="ghost"
                size="icon-sm"
                aria-label="Files"
                className="rounded-full text-zinc-600 hover:bg-zinc-100"
              >
                <FilesIcon className="!size-5" />
              </ShadcnButton>
              <SidebarTrigger />
            </div>
          </div>
          <Button
            variant="primary"
            size="small"
            className="w-full"
            leftIcon={<PlusIcon className="h-4 w-4" weight="bold" />}
          >
            New Chat
          </Button>
        </div>
      </SidebarHeader>

      <SidebarContent className="gap-4 overflow-y-auto px-4 py-4 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        <div className="flex flex-col gap-1">
          {mockSidebarSessions.map((session, index) => {
            const isActive = index === 0;
            return (
              <div
                key={session.id}
                className={cn(
                  "w-full px-3 py-2.5",
                  isActive
                    ? "rounded-lg bg-zinc-100"
                    : "border-b border-b-[#8080800f] last:border-b-0",
                )}
              >
                <ChatSessionBlock
                  title={session.title}
                  updatedAt={session.updated_at}
                  sourcePlatform={session.source_platform}
                  isActive={isActive}
                  chatStatus={session.chat_status}
                />
              </div>
            );
          })}
        </div>
      </SidebarContent>
    </Sidebar>
  );
}
