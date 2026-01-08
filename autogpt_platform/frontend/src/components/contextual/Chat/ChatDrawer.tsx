"use client";

import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { X } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { Drawer } from "vaul";
import { Chat } from "./Chat";
import { useChatDrawer } from "./useChatDrawer";

interface ChatDrawerProps {
  blurBackground?: boolean;
}

export function ChatDrawer({ blurBackground = true }: ChatDrawerProps) {
  const [isMounted, setIsMounted] = useState(false);
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const { isOpen, close } = useChatDrawer();

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    if (isChatEnabled === false && isOpen) {
      close();
    }
  }, [isChatEnabled, isOpen, close]);

  // Don't render on server - vaul drawer accesses document during SSR
  if (!isMounted || isChatEnabled === null || isChatEnabled === false) {
    return null;
  }

  return (
    <Drawer.Root
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          close();
        }
      }}
      direction="right"
      modal={false}
    >
      {blurBackground && isOpen && (
        <div
          onClick={close}
          className="fixed inset-0 z-[45] cursor-pointer animate-in fade-in-0"
          style={{ pointerEvents: "auto" }}
        />
      )}
      <Drawer.Content
        onClick={(e) => e.stopPropagation()}
        onInteractOutside={blurBackground ? close : undefined}
        className={cn(
          "fixed right-0 top-[60px] z-50 flex h-[calc(100vh-60px)] w-1/2 flex-col border-l border-zinc-200 bg-white",
          scrollbarStyles,
        )}
      >
        <Chat
          headerTitle={
            <Drawer.Title className="text-lg font-semibold">
              AutoGPT Copilot
            </Drawer.Title>
          }
          headerActions={
            <button aria-label="Close" onClick={close} className="size-8">
              <X width="1.25rem" height="1.25rem" />
            </button>
          }
        />
      </Drawer.Content>
    </Drawer.Root>
  );
}
