"use client";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/molecules/Sidebar/Sidebar";
import { useReducedMotion } from "framer-motion";
import { Eclipse } from "loading-dev";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { CollapsibleNavGroup } from "../CollapsibleNavGroup/CollapsibleNavGroup";
import { getNewChatHref } from "../RecentChats/helpers";
import { useWorkingExperts } from "./useWorkingExperts";

interface RowProps {
  href: string;
  name: string;
  role: string | null;
  avatarUrl: string | null;
  isOtto?: boolean;
  isActive?: boolean;
  /** How many tasks they are working on right now. */
  workingCount: number;
}

const AVATAR_SIZE = 28;
const WORKING_SIZE = 14;

function ExpertRow({
  href,
  name,
  role,
  avatarUrl,
  isOtto,
  isActive,
  workingCount,
}: RowProps) {
  const prefersReducedMotion = useReducedMotion();
  const working =
    workingCount > 1 ? `Working on ${workingCount} tasks` : "Working on a task";

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        size="shell"
        tooltip={name}
        isActive={isActive}
      >
        <Link href={href}>
          {isOtto ? (
            // The row's gap is 8px (12px on phones); on desktop the avatar
            // sits 2px further off the name.
            <AutopilotAvatar
              size={AVATAR_SIZE}
              shape="soft"
              className="mr-0.5 max-md:mr-0"
            />
          ) : (
            <ExpertAvatar
              name={name}
              avatarUrl={avatarUrl}
              size={AVATAR_SIZE}
              shape="soft"
              className="mr-0.5 max-md:mr-0"
            />
          )}
          <ExpertIdentityDetails
            isOtto={isOtto}
            name={name}
            role={role}
            size="compact"
            areaTone="secondary-soft"
          />
          {workingCount > 0 ? (
            // The same turning mark a working task has in the chat.
            <span
              role="img"
              aria-label={working}
              title={working}
              className="ml-auto flex shrink-0 items-center gap-1 text-foreground-text group-data-[collapsible=icon]:hidden"
            >
              {workingCount > 1 ? (
                <span className="text-caption text-foreground-quiet tabular-nums">
                  {workingCount}
                </span>
              ) : null}
              <Eclipse
                size={WORKING_SIZE}
                playState={prefersReducedMotion ? "paused" : "running"}
              />
            </span>
          ) : null}
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

export function ExpertsNavSection() {
  const { activeExperts, activeExpertIds } = useExpertMap();
  const { workingCount } = useWorkingExperts();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const chatExpert = useCopilotUIStore((s) => s.contextPanelExpert);
  // Whose chat is on screen, so their row is highlighted like a hovered one:
  // an open chat's own expert, or the one a new chat is addressed to. Null is
  // Otto; undefined is any page that is not a chat, where no row is lit.
  const isChatPage = pathname === "/home" || pathname === "/copilot";
  const currentExpertId = !isChatPage
    ? undefined
    : searchParams.get("sessionId")
      ? (chatExpert?.id ?? null)
      : searchParams.get("expertId");

  return (
    <CollapsibleNavGroup label="Experts">
      <SidebarMenu>
        <ExpertRow
          href={getNewChatHref(null, activeExpertIds) ?? "/home"}
          name="Otto"
          role="Head of AI"
          avatarUrl={null}
          isOtto
          isActive={currentExpertId === null}
          workingCount={workingCount(null)}
        />
        {activeExperts.map((expert) => (
          <ExpertRow
            key={expert.id}
            href={getNewChatHref(expert.id, activeExpertIds) ?? "/home"}
            name={expert.name}
            role={expert.jobTitle || expert.role}
            avatarUrl={expert.avatarUrl}
            isActive={currentExpertId === expert.id}
            workingCount={workingCount(expert.id)}
          />
        ))}
      </SidebarMenu>
    </CollapsibleNavGroup>
  );
}
