"use client";

import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  SidebarMenuAction,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { getExpertHref, getPresenceColor, getPresenceLabel } from "./helpers";
import { useSidebarTeamMembers } from "./useSidebarTeamMembers";

// Caps the nested list so the sidebar's overflow-hidden container can never
// clip the trailing "View all" row; the full roster lives on /team.
export const SIDEBAR_TEAM_PREVIEW_COUNT = 3;

export function SidebarTeamMembers() {
  const { isEnabled, members } = useSidebarTeamMembers();

  if (!isEnabled) return null;

  const visibleMembers = members.slice(0, SIDEBAR_TEAM_PREVIEW_COUNT);
  const hasHiddenMembers = members.length > visibleMembers.length;

  return (
    <Collapsible defaultOpen className="group/team">
      {/* Rendered as a menu action so the row's <Link> keeps navigating to
          /team while the chevron only toggles the nested roster. */}
      <CollapsibleTrigger asChild>
        <SidebarMenuAction aria-label="Toggle team members">
          <Icon
            icon={ArrowDown01Icon}
            className="ease-[cubic-bezier(0.33,1,0.68,1)] size-4 text-sidebar-foreground/90 transition-transform duration-200 group-data-[state=open]/team:rotate-180 motion-reduce:transition-none"
          />
        </SidebarMenuAction>
      </CollapsibleTrigger>

      <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
        <SidebarMenuSub>
          {visibleMembers.map((member) => (
            <TeamMemberRow key={member.expert.id} member={member} />
          ))}

          {hasHiddenMembers && (
            <SidebarMenuSubItem>
              <SidebarMenuSubButton asChild className="text-zinc-500">
                <Link href="/team">
                  <span className="truncate">View all ({members.length})</span>
                </Link>
              </SidebarMenuSubButton>
            </SidebarMenuSubItem>
          )}
        </SidebarMenuSub>
      </CollapsibleContent>
    </Collapsible>
  );
}

interface Props {
  member: HomeAgentStatus;
}

function TeamMemberRow({ member }: Props) {
  const { expert, status, detail } = member;
  const label = getPresenceLabel(status);

  return (
    <SidebarMenuSubItem>
      <SidebarMenuSubButton asChild>
        <Link href={getExpertHref(expert.id)}>
          <ExpertAvatar
            name={expert.name}
            avatarUrl={expert.avatar_url}
            size={20}
          />
          <span className="truncate">{expert.name}</span>
          <Tooltip>
            {/* asChild keeps the trigger a span: a button inside the row's
                anchor would be invalid markup and swallow the row's click. */}
            <TooltipTrigger asChild>
              <span
                role="img"
                aria-label={label}
                className={cn(
                  "ml-auto size-2 shrink-0 rounded-full",
                  getPresenceColor(status),
                )}
              />
            </TooltipTrigger>
            {/* Portalled: the sub-button is overflow-hidden and would
                otherwise clip the tooltip down to a sliver. */}
            <TooltipPortal>
              <TooltipContent side="right">{detail || label}</TooltipContent>
            </TooltipPortal>
          </Tooltip>
        </Link>
      </SidebarMenuSubButton>
    </SidebarMenuSubItem>
  );
}
