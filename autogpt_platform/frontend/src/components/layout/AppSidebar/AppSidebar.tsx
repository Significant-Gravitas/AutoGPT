"use client";

import { IconAutoGPTLogo } from "@/components/__legacy__/ui/icons";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  Sparkle,
  TreeStructure,
  Compass,
  Wrench,
  GearSix,
} from "@phosphor-icons/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode } from "react";

interface Props {
  dynamicContent?: ReactNode;
}

export function AppSidebar({ dynamicContent }: Props) {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const pathname = usePathname();
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const { isLoggedIn } = useSupabase();

  const homeHref = isChatEnabled === true ? "/copilot" : "/library";

  const navLinks = [
    isChatEnabled === true
      ? {
          name: "Copilot",
          href: "/copilot",
          icon: Sparkle,
          testId: "sidebar-link-copilot",
        }
      : {
          name: "Library",
          href: "/library",
          icon: TreeStructure,
          testId: "sidebar-link-library",
        },
    ...(isChatEnabled === true
      ? [
          {
            name: "Workflows",
            href: "/library",
            icon: TreeStructure,
            testId: "sidebar-link-workflows",
          },
        ]
      : []),
    {
      name: "Explore",
      href: "/marketplace",
      icon: Compass,
      testId: "sidebar-link-marketplace",
    },
    {
      name: "Builder",
      href: "/build",
      icon: Wrench,
      testId: "sidebar-link-build",
    },
    {
      name: "Settings",
      href: "/profile/settings",
      icon: GearSix,
      testId: "sidebar-link-settings",
    },
  ];

  function isActive(href: string) {
    if (href === homeHref) {
      return pathname === "/" || pathname.startsWith(homeHref);
    }
    return pathname.startsWith(href);
  }

  if (!isLoggedIn) return null;

  return (
    <Sidebar
      variant="sidebar"
      collapsible="icon"
      className="border-r border-zinc-100"
    >
      <SidebarHeader className="border-b border-zinc-100 px-3 py-4">
        <div
          className={cn(
            "flex items-center",
            isCollapsed ? "justify-center" : "justify-between",
          )}
        >
          {!isCollapsed && (
            <Link href={homeHref}>
              <IconAutoGPTLogo className="h-8 w-24" />
            </Link>
          )}
          {!isCollapsed && (
            <div className="flex items-center gap-1">
              <Tooltip>
                <TooltipTrigger asChild>
                  <SidebarTrigger />
                </TooltipTrigger>
                <TooltipContent side="right">Close sidebar</TooltipContent>
              </Tooltip>
            </div>
          )}
        </div>
        {isCollapsed && (
          <div className="mt-2 flex justify-center">
            <Tooltip>
              <TooltipTrigger asChild>
                <SidebarTrigger />
              </TooltipTrigger>
              <TooltipContent side="right">Open sidebar</TooltipContent>
            </Tooltip>
          </div>
        )}
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu className={cn(isCollapsed && "gap-3")}>
              {navLinks.map((link) => (
                <SidebarMenuItem key={link.name}>
                  <SidebarMenuButton
                    asChild
                    isActive={isActive(link.href)}
                    tooltip={link.name}
                    className="py-5 data-[active=true]:bg-violet-50 data-[active=true]:font-normal data-[active=true]:text-violet-700"
                  >
                    <Link href={link.href} data-testid={link.testId}>
                      <link.icon className="!size-5" weight="regular" />
                      <span>{link.name}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarSeparator className="mx-0" />

        {dynamicContent && (
          <SidebarGroup className="flex-1 overflow-hidden">
            {!isCollapsed && (
              <SidebarGroupContent className="h-full overflow-y-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                {dynamicContent}
              </SidebarGroupContent>
            )}
          </SidebarGroup>
        )}
      </SidebarContent>
    </Sidebar>
  );
}
