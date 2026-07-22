"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { AccountMenu } from "@/components/layout/Navbar/components/AccountMenu/AccountMenu";
import { Wallet } from "@/components/layout/Navbar/components/Wallet/Wallet";
import { getAccountMenuItems } from "@/components/layout/Navbar/helpers";
import { SidebarFooter, useSidebar } from "@/components/ui/sidebar";
import {
  Tooltip as SidebarTooltip,
  TooltipContent as SidebarTooltipContent,
  TooltipTrigger as SidebarTooltipTrigger,
} from "@/components/ui/tooltip";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

export function SidebarUserActions() {
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const logoutInProgress = isLogoutInProgress();
  const dynamicMenuItems = getAccountMenuItems(user?.role, true);

  const { data: profile, isLoading: isProfileLoading } = useGetV2GetUserProfile(
    {
      query: {
        select: okData,
        enabled: isLoggedIn && !!user && !logoutInProgress,
        queryKey: ["/api/store/profile", user?.id],
      },
    },
  );

  if (!isLoggedIn) return null;

  const isLoadingProfile = isProfileLoading || isUserLoading;

  const accountMenu = (
    <AccountMenu
      userName={profile?.name || profile?.username}
      userEmail={user?.email}
      avatarSrc={profile?.avatar_url ?? ""}
      menuItemGroups={dynamicMenuItems}
      isLoading={isLoadingProfile}
      newLayout
      side="top"
      align="start"
    />
  );

  return (
    <SidebarFooter className="border-t border-zinc-100 px-4">
      <div className="flex w-full items-center justify-between group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-1">
        {isCollapsed ? (
          <SidebarTooltip>
            <SidebarTooltipTrigger asChild>
              <div>{accountMenu}</div>
            </SidebarTooltipTrigger>
            <SidebarTooltipContent side="right">Account</SidebarTooltipContent>
          </SidebarTooltip>
        ) : (
          accountMenu
        )}
        {profile && (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="group-data-[collapsible=icon]:hidden">
                <Wallet key={profile.username} compact />
              </div>
            </TooltipTrigger>
            <TooltipContent side="top">Credits</TooltipContent>
          </Tooltip>
        )}
      </div>
    </SidebarFooter>
  );
}
