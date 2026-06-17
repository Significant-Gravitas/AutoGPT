"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { IconType } from "@/components/__legacy__/ui/icons";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { PreviewBanner } from "@/components/layout/Navbar/components/PreviewBanner/PreviewBanner";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { AccountMenu } from "./components/AccountMenu/AccountMenu";
import { FeedbackButton } from "./components/FeedbackButton";
import { AgentActivityDropdown } from "./components/AgentActivityDropdown/AgentActivityDropdown";
import { LoginButton } from "./components/LoginButton";
import { MobileNavBar } from "./components/MobileNavbar/MobileNavBar";
import { NavbarLink } from "./components/NavbarLink";
import { NavbarLoading } from "./components/NavbarLoading";
import { Wallet } from "./components/Wallet/Wallet";
import { getAccountMenuItems, loggedInLinks, loggedOutLinks } from "./helpers";

const MOBILE_NAV_ICONS: Readonly<Record<string, IconType>> = {
  "/marketplace": IconType.Marketplace,
  "/build": IconType.Builder,
  "/copilot": IconType.Chat,
  "/library": IconType.Library,
  "/monitor": IconType.Library,
};

function pickMobileNavIcon(href: string): IconType {
  return MOBILE_NAV_ICONS[href] ?? IconType.LayoutDashboard;
}

export function Navbar() {
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const breakpoint = useBreakpoint();
  const isSmallScreen = breakpoint === "sm" || breakpoint === "base";
  const dynamicMenuItems = getAccountMenuItems(user?.role);
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

  // Files is reached via the icon in the CoPilot sidebar (gated by
  // ARTIFACTS_PAGE), not the top nav — it's an entry point, not a product.
  const actualLoggedInLinks = [
    { name: "Home", href: "/copilot" },
    { name: "Agents", href: "/library" },
    ...loggedInLinks,
  ];

  if (isUserLoading) {
    return <NavbarLoading />;
  }

  return (
    <>
      {/* Drives banner-aware offsets in fixed-position siblings (e.g. ChatSidebar)
          that can't read the navbar's natural height through normal layout. */}
      <style>{`:root { --preview-banner-height: ${
        shouldShowPreviewBanner ? "2.25rem" : "0px"
      }; }`}</style>
      <div className="sticky top-0 z-40 w-full">
        {shouldShowPreviewBanner && previewBranchName ? (
          <PreviewBanner branchName={previewBranchName} />
        ) : null}
        <nav
          className="inline-flex w-full items-center border-b border-[#f1f1f1] bg-[#FAFAFA]/80 p-3 backdrop-blur-xl"
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
          <div className="static md:absolute md:left-1/2 md:top-1/2 md:-translate-x-1/2 md:-translate-y-1/2">
            <AutoGPTLogo className="h-auto w-[4.5rem] md:w-[5.5rem]" />
          </div>

          {/* Right section */}
          {isLoggedIn && !isSmallScreen ? (
            <div className="flex flex-1 items-center justify-end gap-4">
              <div className="flex items-center gap-4">
                <FeedbackButton />
                <AgentActivityDropdown />
                {profile && <Wallet key={profile.username} />}
                <AccountMenu
                  userName={profile?.name || profile?.username}
                  userEmail={user?.email}
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
                  items: actualLoggedInLinks.map((link) => ({
                    icon: pickMobileNavIcon(link.href),
                    text: link.name,
                    href: link.href,
                  })),
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
