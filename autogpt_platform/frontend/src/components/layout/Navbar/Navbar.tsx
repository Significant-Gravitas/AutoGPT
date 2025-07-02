import { IconAutoGPTLogo, IconType } from "@/components/ui/icons";
import { ProfileDetails } from "@/lib/autogpt-server-api/types";
import Wallet from "../../agptui/Wallet";
import { AccountMenu } from "./components/AccountMenu/AccountMenu";
import { MobileNavBar } from "./components/MobileNavbar/MobileNavBar";
import { NavbarLink } from "./components/NavbarLink";

import BackendAPI from "@/lib/autogpt-server-api";
import { getServerUser } from "@/lib/supabase/server/getServerUser";
import { LoginButton } from "./components/LoginButton";
import { accountMenuItems, loggedInLinks, loggedOutLinks } from "./helpers";

async function getProfileData() {
  const api = new BackendAPI();
  const profile = await Promise.resolve(api.getStoreProfile());

  return profile;
}

export async function Navbar() {
  const { user } = await getServerUser();
  const isLoggedIn = user !== null;

  let profile: ProfileDetails | null = null;

  if (isLoggedIn) {
    profile = await getProfileData();
  }

  return (
    <>
      <nav className="sticky top-0 z-40 mx-[16px] hidden h-16 items-center rounded-bl-2xl rounded-br-2xl border border-white/50 bg-white/5 p-3 backdrop-blur-[26px] dark:border-gray-700 dark:bg-gray-900 md:inline-flex">
        {/* Left section */}
        <div className="flex flex-1 items-center gap-6">
          {isLoggedIn
            ? loggedInLinks.map((link) => (
                <NavbarLink key={link.name} name={link.name} href={link.href} />
              ))
            : loggedOutLinks.map((link) => (
                <NavbarLink key={link.name} name={link.name} href={link.href} />
              ))}
        </div>

        {/* Centered logo */}
        <div className="absolute left-1/2 top-1/2 h-10 w-[88.87px] -translate-x-1/2 -translate-y-1/2">
          <IconAutoGPTLogo className="h-full w-full" />
        </div>

        {/* Right section */}
        <div className="flex flex-1 items-center justify-end gap-4">
          {isLoggedIn ? (
            <div className="flex items-center gap-4">
              {profile && <Wallet />}
              <AccountMenu
                userName={profile?.username}
                userEmail={profile?.name}
                avatarSrc={profile?.avatar_url}
                menuItemGroups={accountMenuItems}
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
          <div className="fixed right-4 top-4 z-50">
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
                ...accountMenuItems,
              ]}
              userEmail={profile?.name}
              avatarSrc={profile?.avatar_url}
            />
          </div>
        ) : null}
      </>
    </>
  );
}
