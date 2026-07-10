"use client";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
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
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import {
  CaretDownIcon,
  FlowArrowIcon,
  FolderIcon,
  type Icon,
  MagnifyingGlassIcon,
  SparkleIcon,
  SquaresFourIcon,
  StorefrontIcon,
} from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";
import Link from "next/link";
import { tourScenarios } from "../../script/tourScenarios";
import { useTourStore } from "../../tourStore";
import { TourSidebarHeader } from "./components/TourSidebarHeader";

// Visual clone of the logged-in AppSidebar for the public tour demo. Only
// Marketplace navigates; every other destination needs an account, so those
// items render disabled. The "Recent chats" group lists the demo scenarios
// as if they were chat sessions.

function DisabledMenuItem({
  icon: ItemIcon,
  label,
}: {
  icon: Icon;
  label: string;
}) {
  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        aria-disabled="true"
        tooltip={label}
        className="cursor-not-allowed font-normal opacity-50 group-data-[collapsible=icon]:!p-1.5 hover:!bg-transparent [&>svg]:size-5"
      >
        <ItemIcon className="size-5" />
        <span className="truncate">{label}</span>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

function TourSessionsMenu() {
  const activeScenarioId = useTourStore((s) => s.activeScenarioId);
  const setActiveScenario = useTourStore((s) => s.setActiveScenario);
  const clearArtifactPreview = useCopilotUIStore((s) => s.clearArtifactPreview);

  function selectScenario(id: string) {
    clearArtifactPreview();
    setActiveScenario(id);
  }

  return (
    <SidebarMenu>
      {tourScenarios.map((scenario) => (
        <SidebarMenuItem key={scenario.id}>
          <SidebarMenuButton
            isActive={scenario.id === activeScenarioId}
            tooltip={scenario.label}
            onClick={() => selectScenario(scenario.id)}
            className="font-normal data-[active=true]:!bg-zinc-200 data-[active=true]:font-normal hover:!bg-zinc-200"
          >
            <scenario.icon className="size-4 shrink-0" />
            <span className="truncate">{scenario.label}</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      ))}
    </SidebarMenu>
  );
}

export function TourSidebar() {
  const reduceMotion = useReducedMotion();
  const itemVariants = getSidebarItemVariants(!!reduceMotion);

  return (
    <Sidebar
      collapsible="icon"
      className="[&_[data-sidebar=sidebar]]:bg-[#F3F3F4]"
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
                      className="cursor-not-allowed justify-center rounded-lg bg-zinc-800 font-medium text-white opacity-50 group-data-[collapsible=icon]:justify-start hover:!bg-zinc-800 hover:!text-white"
                    >
                      <SparkleIcon className="size-4" />
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
                  <DisabledMenuItem icon={MagnifyingGlassIcon} label="Search" />
                  <DisabledMenuItem icon={SquaresFourIcon} label="Agents" />
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      asChild
                      tooltip="Marketplace"
                      className="font-normal group-data-[collapsible=icon]:!p-1.5 hover:!bg-zinc-200 [&>svg]:size-5"
                    >
                      <Link href="/marketplace">
                        <StorefrontIcon className="size-5" />
                        <span className="truncate">Marketplace</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <DisabledMenuItem icon={FlowArrowIcon} label="Build" />
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </motion.div>

          <motion.div variants={itemVariants}>
            <Collapsible defaultOpen className="group/collapsible">
              <SidebarGroup className="py-1">
                <SidebarGroupLabel asChild className="text-[13px] font-medium">
                  <CollapsibleTrigger>
                    Workspace
                    <CaretDownIcon
                      weight="bold"
                      className="ease-[cubic-bezier(0.33,1,0.68,1)] ml-auto size-4 transition-transform duration-200 group-data-[state=open]/collapsible:rotate-180 motion-reduce:transition-none"
                    />
                  </CollapsibleTrigger>
                </SidebarGroupLabel>
                <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
                  <SidebarGroupContent>
                    <SidebarMenu className="group-data-[collapsible=icon]:gap-1">
                      <DisabledMenuItem icon={FolderIcon} label="Files" />
                    </SidebarMenu>
                  </SidebarGroupContent>
                </CollapsibleContent>
              </SidebarGroup>
            </Collapsible>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="group-data-[collapsible=icon]:hidden"
          >
            <SidebarGroup className="py-1">
              <SidebarGroupLabel className="text-[13px] font-medium">
                Recent chats
              </SidebarGroupLabel>
              <SidebarGroupContent>
                <TourSessionsMenu />
              </SidebarGroupContent>
            </SidebarGroup>
          </motion.div>
        </motion.div>
      </SidebarContent>

      <SidebarRail />
    </Sidebar>
  );
}
