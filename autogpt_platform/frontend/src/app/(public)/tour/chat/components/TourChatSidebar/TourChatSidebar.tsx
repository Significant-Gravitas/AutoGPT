"use client";

import { ChatSessionBlock } from "@/app/(platform)/copilot/components/ChatSessionBlock/ChatSessionBlock";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
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
import { tourChats } from "../../script/tourChats";
import { useTourStore } from "../../tourStore";

export function TourChatSidebar() {
  const activeSessionId = useTourStore((s) => s.activeSessionId);
  const setActiveSession = useTourStore((s) => s.setActiveSession);

  return (
    <Sidebar
      variant="inset"
      collapsible="icon"
      className="!top-0 !h-[100dvh] px-0 [&_[data-sidebar=sidebar]]:border-r [&_[data-sidebar=sidebar]]:border-r-[#80808017]"
    >
      <SidebarHeader className="shrink-0 px-4 pb-3 pt-3">
        <div className="flex flex-col gap-3 px-3">
          <AutoGPTLogo className="mx-auto mb-[10px] h-auto w-24" />
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
          {tourChats.map((chat) => {
            const isActive = chat.id === activeSessionId;
            return (
              <button
                key={chat.id}
                type="button"
                onClick={() => setActiveSession(chat.id)}
                className={cn(
                  "w-full px-3 py-2.5 text-left transition-colors",
                  isActive
                    ? "rounded-lg bg-zinc-100"
                    : "rounded-lg border-b border-b-[#8080800f] last:border-b-0 hover:bg-zinc-50",
                )}
              >
                <ChatSessionBlock
                  title={chat.title}
                  updatedAt={chat.updatedAt}
                  isActive={isActive}
                />
              </button>
            );
          })}
        </div>
      </SidebarContent>
    </Sidebar>
  );
}
