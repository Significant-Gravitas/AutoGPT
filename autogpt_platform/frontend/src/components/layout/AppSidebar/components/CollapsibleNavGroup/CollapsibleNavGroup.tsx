"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
} from "@/components/ui/sidebar";
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
    <Collapsible
      defaultOpen
      open={open}
      onOpenChange={onOpenChange}
      className="group/collapsible"
    >
      <SidebarGroup className="py-1">
        <SidebarGroupLabel
          asChild
          className="text-[13px] font-medium text-zinc-500 group-data-[collapsible=icon]:hidden"
        >
          <CollapsibleTrigger>
            {label}
            <Icon
              icon={ArrowDown01Icon}
              className="ease-[cubic-bezier(0.33,1,0.68,1)] ml-auto size-4 text-sidebar-foreground/90 transition-transform duration-200 group-data-[collapsible=icon]:size-4.5 group-data-[state=open]/collapsible:rotate-180 motion-reduce:transition-none"
            />
          </CollapsibleTrigger>
        </SidebarGroupLabel>
        <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
          <SidebarGroupContent>{children}</SidebarGroupContent>
        </CollapsibleContent>
      </SidebarGroup>
    </Collapsible>
  );
}
