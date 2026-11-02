"use client";

import { tourTarget } from "@/lib/tour-targets";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { AccountMenu } from "@/components/layout/Navbar/components/AccountMenu/AccountMenu";
import { Wallet } from "@/components/layout/Navbar/components/Wallet/Wallet";
import { getAccountMenuItems } from "@/components/layout/Navbar/helpers";
import {
  SidebarFooter,
  useSidebar,
} from "@/components/molecules/Sidebar/Sidebar";
import {
  Tooltip as SidebarTooltip,
  TooltipContent as SidebarTooltipContent,
  TooltipTrigger as SidebarTooltipTrigger,
  TooltipPortal,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useAuth } from "@/lib/auth/hooks/useAuth";

export function SidebarUserActions() {
  const { user, isLoggedIn, isUserLoading } = useAuth();
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const logoutInProgress = isLogoutInProgress();
  const isBillingEnabled = useIsBillingEnabled();
  const dynamicMenuItems = getAccountMenuItems(user?.role, isBillingEnabled);

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
      userHandle={profile?.username}
      userEmail={user?.email}
      avatarSrc={profile?.avatar_url ?? ""}
      menuItemGroups={dynamicMenuItems}
      isLoading={isLoadingProfile}
      side="top"
      align="start"
    />
  );

  return (
    <SidebarFooter
      {...tourTarget("account")}
      className="sticky bottom-0 z-10 mt-auto shrink-0 border-t border-separator bg-sidebar px-4"
    >
      <div className="flex w-full items-center justify-between group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-1">
        {isCollapsed ? (
          <SidebarTooltip inheritDelay>
            <SidebarTooltipTrigger asChild>
              <div>{accountMenu}</div>
            </SidebarTooltipTrigger>
            <TooltipPortal>
              <SidebarTooltipContent side="right">
                Account
              </SidebarTooltipContent>
            </TooltipPortal>
          </SidebarTooltip>
        ) : (
          accountMenu
        )}
        {/* Shown in the collapsed rail too: a new account starts there, and
            a wallet that only appears once the sidebar is expanded is one
            new users never find. */}
        {profile && (
          <div {...tourTarget("wallet")}>
            <Wallet key={profile.username} compact />
          </div>
        )}
      </div>
    </SidebarFooter>
  );
}
