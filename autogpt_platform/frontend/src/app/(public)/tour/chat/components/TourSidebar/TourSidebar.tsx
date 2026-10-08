"use client";

import {
  getSidebarItemVariants,
  sidebarContainerVariants,
} from "@/components/layout/AppSidebar/animations";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import { motion, useReducedMotion } from "motion/react";
import Link from "next/link";
import { useTourStore } from "../../tourStore";
import { TourSidebarHeader } from "./components/TourSidebarHeader";
import { TourUpsellCard } from "./components/TourUpsellCard";
import {
  ArrowDown01Icon,
  FlowIcon,
  Folder01Icon,
  GridViewIcon,
  Search01Icon,
  SparklesIcon,
  Store01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";

function DisabledMenuItem({
  icon,
  label,
}: {
  icon: IconSvgElement;
  label: string;
}) {
  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        aria-disabled="true"
        tooltip={label}
        className="cursor-not-allowed font-normal opacity-50 group-data-[collapsible=icon]:p-1.5! hover:bg-transparent [&>svg]:size-5"
      >
        <Icon icon={icon} className="size-5" />
        <span className="truncate">{label}</span>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

interface Props {
  variant?: "tour" | "marketplace";
}

export function TourSidebar({ variant = "tour" }: Props) {
  const reduceMotion = useReducedMotion();
  const itemVariants = getSidebarItemVariants(!!reduceMotion);
  // In the tour, the end card replaces the sidebar upsell after completion.
  // The marketplace always shows the sidebar card.
  const isDemoComplete = useTourStore((s) => s.isDemoComplete);

  return (
    <Sidebar
      collapsible="icon"
      className="**:data-[sidebar=sidebar]:bg-zinc-100"
    >
      <TourSidebarHeader />

      <SidebarContent className="gap-2 overflow-hidden">
        <motion.div
          variants={sidebarContainerVariants}
          initial="hidden"
          animate="show"
          className="flex min-h-0 flex-1 flex-col gap-2"
        >
          <motion.div variants={itemVariants}>
            <SidebarGroup className="mt-2 py-1 group-data-[collapsible=icon]:mt-0">
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      aria-disabled="true"
                      tooltip="New Task"
                      className="cursor-not-allowed justify-center rounded-lg bg-zinc-800 font-medium text-white opacity-50 group-data-[collapsible=icon]:justify-start hover:bg-zinc-800 hover:text-white"
                    >
                      <Icon icon={SparklesIcon} className="size-4" />
                      <span className="truncate">New Task</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </motion.div>

          <motion.div variants={itemVariants}>
            <SidebarGroup className="mt-2 py-1 group-data-[collapsible=icon]:mt-0">
              <SidebarGroupContent>
                <SidebarMenu className="group-data-[collapsible=icon]:gap-1">
                  <DisabledMenuItem icon={Search01Icon} label="Search" />
                  <DisabledMenuItem icon={GridViewIcon} label="Agents" />
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      render={<Link href="/marketplace" />}
                      tooltip="Marketplace"
                      className="font-normal group-data-[collapsible=icon]:p-1.5! hover:bg-zinc-200 [&>svg]:size-5"
                    >
                      <Icon icon={Store01Icon} className="size-5" />
                      <span className="truncate">Marketplace</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <DisabledMenuItem icon={FlowIcon} label="Build" />
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </motion.div>

          <motion.div variants={itemVariants}>
            <Collapsible defaultOpen className="group/collapsible">
              <SidebarGroup className="py-1">
                <SidebarGroupLabel
                  render={<CollapsibleTrigger />}
                  className="text-[13px] font-medium"
                >
                  Workspace
                  <Icon
                    icon={ArrowDown01Icon}
                    className="ml-auto size-4 transition-transform duration-200 ease-[cubic-bezier(0.33,1,0.68,1)] group-data-open/collapsible:rotate-180 motion-reduce:transition-none"
                  />
                </SidebarGroupLabel>
                <CollapsibleContent className="h-(--collapsible-panel-height) overflow-hidden transition-[height] duration-200 ease-[cubic-bezier(0.33,1,0.68,1)] data-ending-style:h-0 data-starting-style:h-0 motion-reduce:transition-none">
                  <SidebarGroupContent>
                    <SidebarMenu className="group-data-[collapsible=icon]:gap-1">
                      <DisabledMenuItem icon={Folder01Icon} label="Files" />
                    </SidebarMenu>
                  </SidebarGroupContent>
                </CollapsibleContent>
              </SidebarGroup>
            </Collapsible>
          </motion.div>
        </motion.div>
      </SidebarContent>

      {(variant === "marketplace" || !isDemoComplete) && (
        <SidebarFooter className="p-3 group-data-[collapsible=icon]:hidden">
          <TourUpsellCard surface={variant} />
        </SidebarFooter>
      )}

      <SidebarRail />
    </Sidebar>
  );
}
