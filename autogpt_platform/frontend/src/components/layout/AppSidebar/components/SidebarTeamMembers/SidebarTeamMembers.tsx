"use client";

import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import {
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from "@/components/ui/sidebar";
import { MARKETPLACE_EXPERTS_HREF } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import {
  getExpertChatHref,
  getPresenceColor,
  getPresenceLabel,
} from "./helpers";
import { useSidebarTeamMembers } from "./useSidebarTeamMembers";

// Caps the nested list so the sidebar's overflow-hidden container can never
// clip the trailing "Hire" action; the full roster lives on /team.
export const SIDEBAR_TEAM_PREVIEW_COUNT = 5;

export function SidebarTeamMembers() {
  const { isEnabled, members } = useSidebarTeamMembers();

  if (!isEnabled) return null;

  const visibleMembers = members.slice(0, SIDEBAR_TEAM_PREVIEW_COUNT);
  const hasHiddenMembers = members.length > visibleMembers.length;

  return (
    <SidebarMenuSub>
      <SidebarMenuSubItem>
        <SidebarMenuSubButton asChild>
          <Link href="/copilot">
            <ExpertAvatar name={null} avatarUrl={null} size={20} />
            <span className="truncate">Your AI</span>
          </Link>
        </SidebarMenuSubButton>
      </SidebarMenuSubItem>

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

      <SidebarMenuSubItem>
        <SidebarMenuSubButton asChild className="text-zinc-500">
          <Link href={MARKETPLACE_EXPERTS_HREF}>
            <Icon icon={PlusSignIcon} className="size-4" />
            <span className="truncate">Hire</span>
          </Link>
        </SidebarMenuSubButton>
      </SidebarMenuSubItem>
    </SidebarMenuSub>
  );
}

interface Props {
  member: HomeAgentStatus;
}

function TeamMemberRow({ member }: Props) {
  const { expert, status } = member;

  return (
    <SidebarMenuSubItem>
      <SidebarMenuSubButton asChild>
        <Link href={getExpertChatHref(expert.id)}>
          <ExpertAvatar
            name={expert.name}
            avatarUrl={expert.avatar_url}
            size={20}
          />
          <span className="truncate">{expert.name}</span>
          <span
            role="img"
            aria-label={getPresenceLabel(status)}
            className={cn(
              "ml-auto size-2 shrink-0 rounded-full",
              getPresenceColor(status),
            )}
          />
        </Link>
      </SidebarMenuSubButton>
    </SidebarMenuSubItem>
  );
}
