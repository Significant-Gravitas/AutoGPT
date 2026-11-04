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
import { cn } from "@/lib/utils";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { CollapsibleNavGroup } from "../CollapsibleNavGroup/CollapsibleNavGroup";
import { useNavItemClassName } from "../../useNavItemClassName";
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
  const navItemClassName = useNavItemClassName();

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        tooltip={name}
        isActive={isActive}
        className={cn(navItemClassName, "gap-2.5")}
      >
        <Link href={href}>
          {isOtto ? (
            <AutopilotAvatar size={AVATAR_SIZE} className="rounded-lg" />
          ) : (
            <ExpertAvatar
              name={name}
              avatarUrl={avatarUrl}
              size={AVATAR_SIZE}
              className="rounded-lg"
            />
          )}
          <ExpertIdentityDetails
            isOtto={isOtto}
            name={name}
            role={role}
            size="compact"
            areaClassName="opacity-70"
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
      <SidebarMenu className="gap-1">
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
