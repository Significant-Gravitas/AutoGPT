"use client";
import { Sidebar, SidebarHeader, SidebarContent, SidebarFooter, SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { SparkleIcon, PlusIcon, SpinnerGapIcon } from "@phosphor-icons/react";
import { motion } from "framer-motion";
import { useState } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";

export function ChatSidebar() {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const [isCreating, setIsCreating] = useState(false);
  const [, setSessionId] = useQueryState("sessionId", parseAsString);

  async function handleNewChat() {
    if (isCreating) return;
    setIsCreating(true);
    try {
      const response = await postV2CreateSession({
        body: JSON.stringify({}),
      });
      if (response.status === 200 && response.data?.id) {
        setSessionId(response.data.id);
      }
    } finally {
      setIsCreating(false);
    }
  }

  return (
    <Sidebar
      variant="inset"
      collapsible="icon"
      className="!top-[60px] !h-[calc(100vh-60px)]"
    >
      {isCollapsed && <SidebarHeader
        className={cn(
          "flex ",
          isCollapsed
            ? "flex-row items-center justify-between gap-y-4 md:flex-col md:items-start md:justify-start"
            : "flex-row items-center justify-between"
        )}
      >
        <motion.div
          key={isCollapsed ? "header-collapsed" : "header-expanded"}
          className={cn(
            "flex items-center gap-2",
            isCollapsed ? "flex-row md:flex-col-reverse" : "flex-row"
          )}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        >
          {isCollapsed && <div className="bg-secondary border border-neutral-400 h-fit p-1 rounded-3xl">
        <SidebarTrigger />

        </div>}
        </motion.div>
      </SidebarHeader>}
      <SidebarContent className="gap-4 px-2 py-4">
        <div className="flex items-center gap-2">
        <Button
          variant="primary"
          size="icon"
          onClick={handleNewChat}
          disabled={isCreating}
          className={cn(
            "w-full  gap-2 rounded-3xl flex items-center justify-start h-fit px-4 py-2",
            isCollapsed && "justify-center px-1 rounded-3xl "
          )}
        >
          {isCreating ? (
            <SpinnerGapIcon className="h-4 w-4 animate-spin" />
          ) : (
            <PlusIcon className="h-4 w-4" />
          )}
          {!isCollapsed && <span>{isCreating ? "Creating..." : "New Chat"}</span>}
        </Button>
       {!isCollapsed && <div className="bg-secondary border border-neutral-400 h-fit p-1 rounded-3xl">
        <SidebarTrigger />

        </div>}

        </div>
     
      </SidebarContent>
      <SidebarFooter className="px-2">
      </SidebarFooter>
    </Sidebar>
  );
}
