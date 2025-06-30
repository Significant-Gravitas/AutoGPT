import { IconAutoGPTLogo, IconType } from "@/components/ui/icons";
import { ProfileDetails } from "@/lib/autogpt-server-api/types";
import Link from "next/link";
import { MobileNavBar } from "./MobileNavBar";
import { NavbarLink } from "./NavbarLink";
import { ProfilePopoutMenu } from "./ProfilePopoutMenu";
import Wallet from "./Wallet";

import BackendAPI from "@/lib/autogpt-server-api";
import { getServerUser } from "@/lib/supabase/server/getServerUser";
import { SignInIcon } from "@phosphor-icons/react/dist/ssr";
import { Button } from "../atoms/Button/Button";

// Disable theme toggle for now
// import { ThemeToggle } from "./ThemeToggle";

interface NavLink {
  name: string;
  href: string;
}

interface NavbarProps {
  links: NavLink[];
  menuItemGroups: {
    groupName?: string;
    items: {
      icon: IconType;
      text: string;
      href?: string;
      onClick?: () => void;
    }[];
  }[];
}

async function getProfileData() {
  const api = new BackendAPI();
  const profile = await Promise.resolve(api.getStoreProfile());

  return profile;
}

export const Navbar = async ({ links, menuItemGroups }: NavbarProps) => {
  const { user } = await getServerUser();
  const isLoggedIn = user !== null;
  let profile: ProfileDetails | null = null;
  if (isLoggedIn) {
    profile = await getProfileData();
  }

  return (
    <>
      <nav className="sticky top-0 z-40 mx-[16px] hidden h-16 items-center justify-between rounded-bl-2xl rounded-br-2xl border border-white/50 bg-white/5 py-3 pl-6 pr-3 backdrop-blur-[26px] dark:border-gray-700 dark:bg-gray-900 md:inline-flex">
        <div className="flex items-center gap-11">
          <div className="relative h-10 w-[88.87px]">
            <IconAutoGPTLogo className="h-full w-full" />
          </div>
          {links.map((link) => (
            <NavbarLink key={link.name} name={link.name} href={link.href} />
          ))}
        </div>
        {/* Profile section */}
        <div className="flex items-center gap-4">
          {isLoggedIn ? (
            <div className="flex items-center gap-4">
              {profile && <Wallet />}
              <ProfilePopoutMenu
                menuItemGroups={menuItemGroups}
                userName={profile?.username}
                userEmail={profile?.name}
                avatarSrc={profile?.avatar_url}
              />
            </div>
          ) : (
            <Link href="/login">
              <Button
                size="small"
                className="flex items-center justify-end space-x-2"
                leftIcon={<SignInIcon className="h-5 w-5" />}
                variant="secondary"
              >
                Log In
              </Button>
            </Link>
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
                  items: links.map((link) => ({
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
                ...menuItemGroups,
              ]}
              userEmail={profile?.name}
              avatarSrc={profile?.avatar_url}
            />
          </div>
        ) : (
          <Link
            href="/login"
            className="fixed right-4 top-4 z-50 mt-4 inline-flex h-8 items-center justify-end rounded-lg pr-4 md:hidden"
          >
            <Button
              size="small"
              className="flex items-center justify-end space-x-2"
              leftIcon={<SignInIcon className="h-5 w-5" />}
              variant="secondary"
            >
              Log In
            </Button>
          </Link>
        )}
      </>
    </>
  );
};
