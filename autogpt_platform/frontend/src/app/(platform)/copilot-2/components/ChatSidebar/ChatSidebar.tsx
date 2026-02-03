"use client";
import { Sidebar, SidebarHeader, SidebarContent, SidebarFooter, SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { SparkleIcon, PlusIcon } from "@phosphor-icons/react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

export function ChatSidebar() {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";

  function handleNewChat() {
    // TODO: Implement new chat creation
  }

  return (
    <Sidebar
      variant="inset"
      collapsible="icon"
      className="!top-[60px] !h-[calc(100vh-60px)]"
    >
      <SidebarHeader
        className={cn(
          "flex md:pt-3.5",
          isCollapsed
            ? "flex-row items-center justify-between gap-y-4 md:flex-col md:items-start md:justify-start"
            : "flex-row items-center justify-between"
        )}
      >
        <a href="#" className="flex items-center gap-2">
          <SparkleIcon className="h-8 w-8" />
          {!isCollapsed && (
            <span className="font-semibold text-black dark:text-white">
              Acme
            </span>
          )}
        </a>

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
          <SidebarTrigger />
        </motion.div>
      </SidebarHeader>
      <SidebarContent className="gap-4 px-2 py-4">
        <Button
          onClick={handleNewChat}
          className={cn(
            "w-full justify-start gap-2",
            isCollapsed && "justify-center px-2"
          )}
        >
          <PlusIcon className="h-4 w-4" />
          {!isCollapsed && <span>New Chat</span>}
        </Button>
      </SidebarContent>
      <SidebarFooter className="px-2">
      </SidebarFooter>
    </Sidebar>
  );
}
