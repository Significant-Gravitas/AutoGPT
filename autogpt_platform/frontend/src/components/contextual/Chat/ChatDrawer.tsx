"use client";

import { Button } from "@/components/__legacy__/ui/button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { X } from "@phosphor-icons/react";
import { useEffect } from "react";
import { Drawer } from "vaul";
import { Chat } from "./Chat";
import { useChatDrawer } from "./useChatDrawer";

interface ChatDrawerProps {
  blurBackground?: boolean;
}

export function ChatDrawer({ blurBackground = true }: ChatDrawerProps) {
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const { isOpen, close } = useChatDrawer();

  useEffect(() => {
    if (isChatEnabled === false && isOpen) {
      close();
    }
  }, [isChatEnabled, isOpen, close]);

  if (isChatEnabled === null || isChatEnabled === false) {
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
      <Drawer.Portal>
        {blurBackground && isOpen && (
          <div
            onClick={close}
            className="fixed inset-0 z-[45] cursor-pointer bg-black/10 backdrop-blur-sm animate-in fade-in-0"
            style={{ pointerEvents: "auto" }}
          />
        )}
        <Drawer.Content
          onClick={(e) => e.stopPropagation()}
          onInteractOutside={blurBackground ? close : undefined}
          className={cn(
            "fixed right-0 top-0 z-50 flex h-full w-1/2 flex-col border-l border-zinc-200 bg-white",
            scrollbarStyles,
          )}
        >
          <Chat
            headerTitle={
              <Drawer.Title className="text-xl font-semibold">
                Chat
              </Drawer.Title>
            }
            headerActions={
              <Button
                variant="link"
                aria-label="Close"
                onClick={close}
                className="!focus-visible:ring-0 p-0"
              >
                <X width="1.5rem" />
              </Button>
            }
          />
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
