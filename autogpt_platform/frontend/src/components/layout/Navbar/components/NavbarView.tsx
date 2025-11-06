"use client";

import { useMemo } from "react";
import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { IconAutoGPTLogo, IconType } from "@/components/__legacy__/ui/icons";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { getAccountMenuItems, loggedInLinks, loggedOutLinks } from "../helpers";
import { AccountMenu } from "./AccountMenu/AccountMenu";
import { AgentActivityDropdown } from "./AgentActivityDropdown/AgentActivityDropdown";
import { LoginButton } from "./LoginButton";
import { MobileNavBar } from "./MobileNavbar/MobileNavBar";
import { NavbarLink } from "./NavbarLink";
import { Wallet } from "./Wallet/Wallet";
import { useGetFlag, Flag } from "@/services/feature-flags/use-get-flag";
interface NavbarViewProps {
  isLoggedIn: boolean;
}

export const NavbarView = ({ isLoggedIn }: NavbarViewProps) => {
  const { user } = useSupabase();
  const breakpoint = useBreakpoint();
  const isSmallScreen = breakpoint === "sm" || breakpoint === "base";
  const dynamicMenuItems = getAccountMenuItems(user?.role);
  const isChatEnabled = useGetFlag(Flag.CHAT);

  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
      enabled: isLoggedIn,
    },
  });

  const linksWithChat = useMemo(() => {
    const chatLink = { name: "Chat", href: "/chat" };
    return isChatEnabled ? [...loggedInLinks, chatLink] : loggedInLinks;
  }, [isChatEnabled]);

  return (
    <>
      <nav className="sticky top-0 z-40 inline-flex h-[60px] w-full items-center border border-white/50 bg-[#f3f4f6]/20 p-3 backdrop-blur-[26px]">
        {/* Left section */}
        {!isSmallScreen ? (
          <div className="flex flex-1 items-center gap-3 gap-5">
            {isLoggedIn
              ? linksWithChat.map((link) => (
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
                  items: linksWithChat.map((link) => ({
                    icon:
                      link.name === "Marketplace"
                        ? IconType.Marketplace
                        : link.name === "Library"
                          ? IconType.Library
                          : link.name === "Build"
                            ? IconType.Builder
                            : link.name === "Chat"
                              ? IconType.Chat
                              : link.name === "Monitor"
                                ? IconType.Library
                                : IconType.LayoutDashboard,
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
};
