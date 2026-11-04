"use client";

import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/molecules/Sidebar/Sidebar";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { CollapsibleNavGroup } from "../CollapsibleNavGroup/CollapsibleNavGroup";
import { getNewChatHref } from "../RecentChats/helpers";

interface RowProps {
  href: string;
  name: string;
  role: string | null;
  avatarUrl: string | null;
  isOtto?: boolean;
  isActive?: boolean;
}

const AVATAR_SIZE = 28;

function ExpertRow({
  href,
  name,
  role,
  avatarUrl,
  isOtto,
  isActive,
}: RowProps) {
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
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

export function ExpertsNavSection() {
  const { activeExperts, activeExpertIds } = useExpertMap();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const currentExpertId =
    pathname === "/home" ? searchParams.get("expertId") : null;

  return (
    <CollapsibleNavGroup label="Experts">
      <SidebarMenu>
        <ExpertRow
          href={getNewChatHref(null, activeExpertIds) ?? "/home"}
          name="Otto"
          role="Head of AI"
          avatarUrl={null}
          isOtto
        />
        {activeExperts.map((expert) => (
          <ExpertRow
            key={expert.id}
            href={getNewChatHref(expert.id, activeExpertIds) ?? "/home"}
            name={expert.name}
            role={expert.jobTitle || expert.role}
            avatarUrl={expert.avatarUrl}
            isActive={currentExpertId === expert.id}
          />
        ))}
      </SidebarMenu>
    </CollapsibleNavGroup>
  );
}
