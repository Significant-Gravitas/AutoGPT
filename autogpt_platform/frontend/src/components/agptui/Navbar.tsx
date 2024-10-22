import * as React from "react";
import Link from "next/link";
import { ProfilePopoutMenu } from "./ProfilePopoutMenu";
import { IconType, IconLogIn } from "../ui/icons";
import { MobileNavBar } from "./MobileNavBar";
import { Button } from "./Button";
interface NavLink {
  name: string;
  href: string;
}

interface NavbarProps {
  isLoggedIn: boolean;
  userName?: string;
  links: NavLink[];
  activeLink: string;
  avatarSrc?: string;
  userEmail?: string;
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

export const Navbar: React.FC<NavbarProps> = ({
  isLoggedIn,
  userName,
  links,
  activeLink,
  avatarSrc,
  userEmail,
  menuItemGroups,
}) => {
  return (
    <>
      <nav className="hidden h-[5.5rem] w-full items-center justify-between border border-black/10 bg-[#f0f0f0] px-16 md:flex">
        <div className="flex items-center space-x-10">
          {links.map((link) => (
            <div key={link.name} className="relative">
              <Link href={link.href}>
                <div
                  className={`text-[${activeLink === link.href ? "#272727" : "#474747"}] font-neue text-2xl font-medium leading-9 tracking-tight`}
                >
                  {link.name}
                </div>
              </Link>
              {activeLink === link.href && (
                <div className="absolute bottom-[-30px] left-[-10px] h-1.5 w-full bg-[#282828]" />
              )}
            </div>
          ))}
        </div>
        {/* Profile section */}
        {isLoggedIn ? (
          <ProfilePopoutMenu
            menuItemGroups={menuItemGroups}
            userName={userName}
            userEmail={userEmail}
            avatarSrc={avatarSrc}
          />
        ) : (
          <Link href="/login">
            <Button
              variant="default"
              size="sm"
              className="flex items-center justify-end space-x-2"
            >
              <IconLogIn className="h-5 w-5" />
              <span>Log In</span>
            </Button>
          </Link>
        )}
      </nav>
      {/* Mobile Navbar */}
      <>
        {isLoggedIn ? (
          <MobileNavBar
            userName={userName}
            activeLink={activeLink}
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
                          : IconType.LayoutDashboard,
                  text: link.name,
                  href: link.href,
                })),
              },
              ...menuItemGroups,
            ]}
            userEmail={userEmail}
            avatarSrc={avatarSrc}
          />
        ) : (
          <Link
            href="/login"
            className="z-50 mt-4 inline-flex h-8 w-screen items-center justify-end rounded-lg pr-4 md:hidden"
          >
            <Button
              variant="default"
              size="sm"
              className="flex items-center space-x-2"
            >
              <IconLogIn className="h-5 w-5" />
              <span>Log In</span>
            </Button>
          </Link>
        )}
      </>
    </>
  );
};
