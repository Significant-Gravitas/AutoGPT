"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { IconAutoGPTLogo, IconType } from "@/components/__legacy__/ui/icons";
import { PreviewBanner } from "@/components/layout/Navbar/components/PreviewBanner/PreviewBanner";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { AccountMenu } from "./components/AccountMenu/AccountMenu";
import { AgentActivityDropdown } from "./components/AgentActivityDropdown/AgentActivityDropdown";
import { LoginButton } from "./components/LoginButton";
import { MobileNavBar } from "./components/MobileNavbar/MobileNavBar";
import { NavbarLink } from "./components/NavbarLink";
import { NavbarLoading } from "./components/NavbarLoading";
import { Wallet } from "./components/Wallet/Wallet";
import { getAccountMenuItems, loggedInLinks, loggedOutLinks } from "./helpers";

export function Navbar() {
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const breakpoint = useBreakpoint();
  const isSmallScreen = breakpoint === "sm" || breakpoint === "base";
  const dynamicMenuItems = getAccountMenuItems(user?.role);
  const isChatEnabled = useGetFlag(Flag.CHAT);
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
    ...(isChatEnabled === true ? [{ name: "Tasks", href: "/library" }] : []),
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
          className="border-zinc-[#EFEFF0] inline-flex w-full items-center border border-[#EFEFF0] bg-[#F3F4F6]/20 p-3 backdrop-blur-[26px]"
          style={{ height: NAVBAR_HEIGHT_PX }}
        >
          {/* Left section */}
          {!isSmallScreen ? (
            <div className="flex flex-1 items-center gap-5">
              {isLoggedIn
                ? actualLoggedInLinks.map((link) => (
                    <NavbarLink
                      key={link.name}
                      name={link.name}
                      href={link.href}
                    />
                  ))
                : loggedOutLinks.map((link) => (
                    <NavbarLink
                      key={link.name}
                      name={link.name}
                      href={link.href}
                    />
                  ))}
            </div>
          ) : null}

          {/* Centered logo */}
          <div className="static h-auto w-[4.5rem] md:absolute md:left-1/2 md:top-1/2 md:w-[5.5rem] md:-translate-x-1/2 md:-translate-y-1/2">
            <IconAutoGPTLogo className="h-full w-full" />
          </div>

          {/* Right section */}
          {isLoggedIn && !isSmallScreen ? (
            <div className="flex flex-1 items-center justify-end gap-4">
              <div className="flex items-center gap-4">
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
            </div>
          ) : !isLoggedIn ? (
            <div className="flex w-full items-center justify-end">
              <LoginButton />
            </div>
          ) : null}
          {/* <ThemeToggle /> */}
        </nav>
      </div>
      {/* Mobile Navbar - Adjust positioning */}
      <>
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
    </>
  );
}
