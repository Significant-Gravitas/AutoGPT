"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { IconAutoGPTLogo, IconType } from "@/components/__legacy__/ui/icons";
import { PreviewBanner } from "@/components/layout/Navbar/components/PreviewBanner/PreviewBanner";
import { useSidebar } from "@/components/ui/sidebar";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { AccountMenu } from "./components/AccountMenu/AccountMenu";
import { FeedbackButton } from "./components/FeedbackButton";
import { AgentActivityDropdown } from "./components/AgentActivityDropdown/AgentActivityDropdown";
import { LoginButton } from "./components/LoginButton";
import { MobileNavBar } from "./components/MobileNavbar/MobileNavBar";
import { NavbarLoading } from "./components/NavbarLoading";
import { Wallet } from "./components/Wallet/Wallet";
import { getAccountMenuItems, loggedInLinks } from "./helpers";

export function Navbar() {
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const breakpoint = useBreakpoint();
  const isSmallScreen = breakpoint === "sm" || breakpoint === "base";
  const dynamicMenuItems = getAccountMenuItems(user?.role);
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const { state: sidebarState } = useSidebar();
  const isSidebarCollapsed = sidebarState === "collapsed";
  const previewBranchName = environment.getPreviewStealingDev();
  const logoutInProgress = isLogoutInProgress();

  const { data: profile, isLoading: isProfileLoading } = useGetV2GetUserProfile(
    {
      query: {
        select: okData,
        enabled: isLoggedIn && !!user && !logoutInProgress,
        // Include user ID in query key to ensure cache invalidation when user changes
        queryKey: ["/api/store/profile", user?.id],
      },
    },
  );

  const isLoadingProfile = isProfileLoading || isUserLoading;

  const shouldShowPreviewBanner = Boolean(isLoggedIn && previewBranchName);

  const homeHref = isChatEnabled === true ? "/copilot" : "/library";

  const actualLoggedInLinks = [
    { name: "Home", href: homeHref },
    ...(isChatEnabled === true ? [{ name: "Agents", href: "/library" }] : []),
    ...loggedInLinks,
  ];

  if (isUserLoading) {
    return <NavbarLoading />;
  }

  return (
    <>
      <div className="sticky top-0 z-40 w-full">
        {shouldShowPreviewBanner && previewBranchName ? (
          <PreviewBanner branchName={previewBranchName} />
        ) : null}
        <nav
          className="relative inline-flex w-full items-center justify-end border-b border-zinc-100 bg-[#FAFAFA]/80 p-3 backdrop-blur-xl"
          style={{ height: NAVBAR_HEIGHT_PX }}
        >
          {isSidebarCollapsed && (
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
              <IconAutoGPTLogo className="h-8 w-24" />
            </div>
          )}
          {/* Right section */}
          {isLoggedIn && !isSmallScreen ? (
            <div className="flex items-center gap-4">
              <FeedbackButton />
              <AgentActivityDropdown />
              {profile && <Wallet key={profile.username} />}
              <AccountMenu
                userName={profile?.username}
                userEmail={profile?.name}
                avatarSrc={profile?.avatar_url ?? ""}
                menuItemGroups={dynamicMenuItems}
                isLoading={isLoadingProfile}
              />
            </div>
          ) : !isLoggedIn ? (
            <LoginButton />
          ) : null}
          {/* <ThemeToggle /> */}
        </nav>
      </div>
      {/* Mobile Navbar - Adjust positioning */}
      {isLoggedIn && isSmallScreen ? (
        <div className="fixed right-0 top-2 z-50 flex items-center gap-0">
          <Wallet />
          <MobileNavBar
            userName={profile?.username}
            menuItemGroups={[
              {
                groupName: "Navigation",
                items: actualLoggedInLinks
                  .map((link) => {
                    return {
                      icon:
                        link.href === "/marketplace"
                          ? IconType.Marketplace
                          : link.href === "/build"
                            ? IconType.Builder
                            : link.href === "/copilot"
                              ? IconType.Chat
                              : link.href === "/library"
                                ? IconType.Library
                                : link.href === "/monitor"
                                  ? IconType.Library
                                  : IconType.LayoutDashboard,
                      text: link.name,
                      href: link.href,
                    };
                  })
                  .filter((item) => item !== null) as Array<{
                  icon: IconType;
                  text: string;
                  href: string;
                }>,
              },
              ...dynamicMenuItems,
            ]}
            userEmail={profile?.name}
            avatarSrc={profile?.avatar_url ?? ""}
          />
        </div>
      ) : null}
    </>
  );
}
