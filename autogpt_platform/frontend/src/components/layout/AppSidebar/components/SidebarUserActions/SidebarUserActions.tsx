"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { UsageIndicator } from "@/app/(platform)/PlatformChrome/components/UsageIndicator/UsageIndicator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { AccountMenu } from "@/components/layout/Navbar/components/AccountMenu/AccountMenu";
import { AgentActivityDropdown } from "@/components/layout/Navbar/components/AgentActivityDropdown/AgentActivityDropdown";
import { Wallet } from "@/components/layout/Navbar/components/Wallet/Wallet";
import { getAccountMenuItems } from "@/components/layout/Navbar/helpers";
import { SidebarFooter } from "@/components/ui/sidebar";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

export function SidebarUserActions() {
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const logoutInProgress = isLogoutInProgress();
  const dynamicMenuItems = getAccountMenuItems(user?.role);

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

  return (
    <SidebarFooter className="border-t border-zinc-100 px-4 shadow-[0_-2px_8px_rgba(0,0,0,0.04)]">
      <div className="flex w-full items-center justify-between group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="[&_button:hover]:bg-zinc-100 [&_button[data-state=open]]:bg-zinc-100 [&_button]:flex [&_button]:h-8 [&_button]:w-8 [&_button]:items-center [&_button]:justify-center [&_button]:rounded-lg [&_button]:bg-transparent [&_button]:p-0 [&_button]:transition-colors [&_svg]:!size-5">
              <AgentActivityDropdown />
            </div>
          </TooltipTrigger>
          <TooltipContent side="top">Agent activity</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <div>
              <UsageIndicator />
            </div>
          </TooltipTrigger>
          <TooltipContent side="top">Today&apos;s usage</TooltipContent>
        </Tooltip>
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
        <Tooltip>
          <TooltipTrigger asChild>
            <div>
              <AccountMenu
                userName={profile?.name || profile?.username}
                userEmail={user?.email}
                avatarSrc={profile?.avatar_url ?? ""}
                menuItemGroups={dynamicMenuItems}
                isLoading={isLoadingProfile}
              />
            </div>
          </TooltipTrigger>
          <TooltipContent side="top">Account</TooltipContent>
        </Tooltip>
      </div>
    </SidebarFooter>
  );
}
