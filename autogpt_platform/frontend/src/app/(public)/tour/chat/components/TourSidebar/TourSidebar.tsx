"use client";

import {
  getSidebarItemVariants,
  sidebarContainerVariants,
} from "@/components/layout/AppSidebar/animations";
import {
  CollapsibleRoot,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/molecules/Collapsible/Collapsible";
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
  useSidebar,
} from "@/components/molecules/Sidebar/Sidebar";
import { motion, useReducedMotion } from "framer-motion";
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
import { Text } from "@/components/atoms/Text/Text";
import { CookieSettingsLink } from "@/components/molecules/CookieSettingsLink/CookieSettingsLink";

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
        // oxlint-disable-next-line design-system/no-restyle -- 20px icons need the tighter collapsed padding; SidebarMenuButton has no size for it
        className="cursor-not-allowed group-data-[collapsible=icon]:!p-1.5 [&>svg]:size-5"
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

// Matches the mobile sidebar sheet's closing animation
// (`data-[state=closed]:duration-200` in `@/components/molecules/Sidebar/Sidebar`).
const SHEET_CLOSE_MS = 200;

export function TourSidebar({ variant = "tour" }: Props) {
  const reduceMotion = useReducedMotion();
  const itemVariants = getSidebarItemVariants(!!reduceMotion);
  // In the tour, the end card replaces the sidebar upsell after completion.
  // The marketplace always shows the sidebar card.
  const isDemoComplete = useTourStore((s) => s.isDemoComplete);
  const { isMobile, openMobile, setOpenMobile } = useSidebar();

  // On phones the sidebar is a modal sheet, which blocks clicks and focus
  // outside itself, Cookiebot's dialog included: close it first.
  const isSheetOpen = isMobile && openMobile;
  async function closeMobileSheet() {
    setOpenMobile(false);
    await new Promise((resolve) => setTimeout(resolve, SHEET_CLOSE_MS));
  }

  return (
    <Sidebar
      collapsible="icon"
      // oxlint-disable-next-line design-system/no-restyle -- the tour sidebar's grey panel; Sidebar has no variant for it
      className="[&_[data-sidebar=sidebar]]:bg-surface-hover"
    >
      <TourSidebarHeader />

      <SidebarContent className="overflow-hidden">
        <motion.div
          variants={sidebarContainerVariants}
          initial="hidden"
          animate="show"
          className="flex min-h-0 flex-1 flex-col gap-2"
        >
          <motion.div variants={itemVariants}>
            {/* oxlint-disable-next-line design-system/no-restyle -- the copilot sidebar's tighter group padding; SidebarGroup has no size for it */}
            <SidebarGroup className="mt-2 py-1 group-data-[collapsible=icon]:mt-0">
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      aria-disabled="true"
                      tooltip="New Task"
                      // oxlint-disable-next-line design-system/no-restyle -- the copilot's dark New Task button; SidebarMenuButton has no primary variant
                      className="cursor-not-allowed justify-center rounded-lg bg-action font-medium text-action-foreground group-data-[collapsible=icon]:justify-start"
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
            {/* oxlint-disable-next-line design-system/no-restyle -- the copilot sidebar's tighter group padding; SidebarGroup has no size for it */}
            <SidebarGroup className="mt-2 py-1 group-data-[collapsible=icon]:mt-0">
              <SidebarGroupContent>
                {/* oxlint-disable-next-line design-system/no-restyle -- the copilot sidebar's collapsed item gap; SidebarMenu has no size for it */}
                <SidebarMenu className="group-data-[collapsible=icon]:gap-1">
                  <DisabledMenuItem icon={Search01Icon} label="Search" />
                  <DisabledMenuItem icon={GridViewIcon} label="Workflows" />
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      asChild
                      tooltip="Marketplace"
                      // oxlint-disable-next-line design-system/no-restyle -- the tour sidebar's darker hover and 20px-icon collapsed padding; SidebarMenuButton has no variant for them
                      className="group-data-[collapsible=icon]:!p-1.5 hover:!bg-surface-active [&>svg]:size-5"
                    >
                      <Link href="/marketplace">
                        <Icon icon={Store01Icon} className="size-5" />
                        <span className="truncate">Marketplace</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <DisabledMenuItem icon={FlowIcon} label="Build" />
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </motion.div>

          <motion.div variants={itemVariants}>
            <CollapsibleRoot defaultOpen className="group/collapsible">
              {/* oxlint-disable-next-line design-system/no-restyle -- the copilot sidebar's tighter group padding; SidebarGroup has no size for it */}
              <SidebarGroup className="py-1">
                <SidebarGroupLabel asChild>
                  <CollapsibleTrigger>
                    <Text
                      as="span"
                      variant="body-sm-medium-inherit"
                      tone="inherit"
                    >
                      Workspace
                    </Text>
                    <span className="ml-auto flex shrink-0 transition-transform duration-200 group-data-[state=open]/collapsible:rotate-180 motion-reduce:transition-none">
                      <Icon icon={ArrowDown01Icon} className="size-4" />
                    </span>
                  </CollapsibleTrigger>
                </SidebarGroupLabel>
                {/* oxlint-disable-next-line design-system/no-restyle -- Radix sets the open state and height on this element, so the collapse animation can't move to a wrapper */}
                <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
                  <SidebarGroupContent>
                    {/* oxlint-disable-next-line design-system/no-restyle -- the copilot sidebar's collapsed item gap; SidebarMenu has no size for it */}
                    <SidebarMenu className="group-data-[collapsible=icon]:gap-1">
                      <DisabledMenuItem icon={Folder01Icon} label="Files" />
                    </SidebarMenu>
                  </SidebarGroupContent>
                </CollapsibleContent>
              </SidebarGroup>
            </CollapsibleRoot>
          </motion.div>
        </motion.div>
      </SidebarContent>

      {/* oxlint-disable-next-line design-system/no-restyle -- the upsell card's 12px inset; SidebarFooter has no size for it */}
      <SidebarFooter className="p-3 group-data-[collapsible=icon]:hidden">
        {(variant === "marketplace" || !isDemoComplete) && (
          <TourUpsellCard surface={variant} />
        )}
        <CookieSettingsLink
          className="self-center"
          beforeOpen={isSheetOpen ? closeMobileSheet : undefined}
        />
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
