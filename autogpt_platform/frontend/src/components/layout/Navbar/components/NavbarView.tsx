"use client";
import { IconAutoGPTLogo, IconType } from "@/components/__legacy__/ui/icons";
import Wallet from "../../../__legacy__/Wallet";
import { AccountMenu } from "./AccountMenu/AccountMenu";
import { LoginButton } from "./LoginButton";
import { MobileNavBar } from "./MobileNavbar/MobileNavBar";
import { NavbarLink } from "./NavbarLink";
import { getAccountMenuItems, loggedInLinks, loggedOutLinks } from "../helpers";
import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { AgentActivityDropdown } from "./AgentActivityDropdown/AgentActivityDropdown";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

interface NavbarViewProps {
  isLoggedIn: boolean;
}

export const NavbarView = ({ isLoggedIn }: NavbarViewProps) => {
  const { user } = useSupabase();
  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
      enabled: isLoggedIn,
    },
  });

  const dynamicMenuItems = getAccountMenuItems(user?.role);

  return (
    <>
      <nav className="sticky top-0 z-40 inline-flex h-16 w-full items-center border border-white/50 bg-[#f3f4f6]/20 p-3 backdrop-blur-[26px]">
        {/* Left section */}
        <div className="hidden flex-1 items-center gap-3 md:flex md:gap-5">
          {isLoggedIn
            ? loggedInLinks.map((link) => (
                <NavbarLink key={link.name} name={link.name} href={link.href} />
              ))
            : loggedOutLinks.map((link) => (
                <NavbarLink key={link.name} name={link.name} href={link.href} />
              ))}
        </div>

        {/* Centered logo */}
        <div className="absolute left-16 top-1/2 h-auto w-[5.5rem] -translate-x-1/2 -translate-y-1/2 md:left-1/2">
          <IconAutoGPTLogo className="h-full w-full" />
        </div>

        {/* Right section */}
        <div className="hidden flex-1 items-center justify-end gap-4 md:flex">
          {isLoggedIn ? (
            <div className="flex items-center gap-4">
              <AgentActivityDropdown />
              {profile && <Wallet />}
              <AccountMenu
                userName={profile?.username}
                userEmail={profile?.name}
                avatarSrc={profile?.avatar_url ?? ""}
                menuItemGroups={dynamicMenuItems}
              />
            </div>
          ) : (
            <LoginButton />
          )}
          {/* <ThemeToggle /> */}
        </div>
      </nav>
      {/* Mobile Navbar - Adjust positioning */}
      <>
        {isLoggedIn ? (
          <div className="fixed right-0 top-2 z-50 flex items-center gap-0 md:hidden">
            <Wallet />
            <MobileNavBar
              userName={profile?.username}
              menuItemGroups={[
                {
                  groupName: "Navigation",
                  items: loggedInLinks.map((link) => ({
                    icon:
                      link.name === "Marketplace"
                        ? IconType.Marketplace
                        : link.name === "Library"
                          ? IconType.Library
                          : link.name === "Build"
                            ? IconType.Builder
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
