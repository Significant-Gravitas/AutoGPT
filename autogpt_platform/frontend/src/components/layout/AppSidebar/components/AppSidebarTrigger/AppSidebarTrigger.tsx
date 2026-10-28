"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Button } from "@/components/atoms/Button/Button";
import { useSidebar } from "@/components/molecules/Sidebar/Sidebar";
import { tourTarget } from "@/lib/tour-targets";
import { SidebarLeftIcon } from "@hugeicons/core-free-icons";
import { createContext, useContext, type ReactNode } from "react";

// The copilot page mounts a SidebarProvider of its own, so a plain
// SidebarTrigger inside the chat toggles that one instead of the app sidebar.
// PlatformChrome hands the app sidebar's toggle down through this context.
const AppSidebarToggleContext = createContext<(() => void) | null>(null);

export function AppSidebarToggleProvider({
  children,
}: {
  children: ReactNode;
}) {
  const { toggleSidebar } = useSidebar();

  return (
    <AppSidebarToggleContext.Provider value={toggleSidebar}>
      {children}
    </AppSidebarToggleContext.Provider>
  );
}

export function useAppSidebarToggle() {
  return useContext(AppSidebarToggleContext);
}

interface Props {
  className?: string;
}

export function AppSidebarTrigger({ className }: Props) {
  const toggleSidebar = useAppSidebarToggle();

  if (!toggleSidebar) return null;

  return (
    <Button
      variant="compact-ghost"
      size="compact-icon"
      data-sidebar="trigger"
      // The phone's way into the sidebar inside a chat: the walkthrough and
      // the tips point here when the sidebar item itself is in the sheet.
      {...tourTarget("menu")}
      className={className}
      onClick={toggleSidebar}
    >
      <Icon icon={SidebarLeftIcon} className="!size-5" />
      <span className="sr-only">Toggle Sidebar</span>
    </Button>
  );
}
