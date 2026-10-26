"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  CollapsibleRoot,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/molecules/Collapsible/Collapsible";
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
} from "@/components/molecules/Sidebar/Sidebar";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";

interface Props {
  label: string;
  children: ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function CollapsibleNavGroup({
  label,
  children,
  open,
  onOpenChange,
}: Props) {
  return (
    <CollapsibleRoot
      defaultOpen
      open={open}
      onOpenChange={onOpenChange}
      className="group/collapsible"
    >
      <SidebarGroup className="py-1">
        <SidebarGroupLabel
          asChild
          className="leading-inherit text-foreground-tertiary group-data-[collapsible=icon]:hidden max-md:h-11"
        >
          <CollapsibleTrigger>
            <Text
              as="span"
              variant="body-sm-medium-inherit-touch"
              tone="inherit"
            >
              {label}
            </Text>
            <Icon
              icon={ArrowDown01Icon}
              className="ease-[cubic-bezier(0.33,1,0.68,1)] ml-auto size-4 text-sidebar-foreground/90 transition-transform duration-200 group-data-[collapsible=icon]:size-4.5 group-data-[state=open]/collapsible:rotate-180 motion-reduce:transition-none max-md:size-5"
            />
          </CollapsibleTrigger>
        </SidebarGroupLabel>
        <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
          <SidebarGroupContent>{children}</SidebarGroupContent>
        </CollapsibleContent>
      </SidebarGroup>
    </CollapsibleRoot>
  );
}
