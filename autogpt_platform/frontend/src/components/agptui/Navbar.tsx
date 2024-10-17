import * as React from "react";
import Link from "next/link";
import { ProfilePopoutMenu } from "./ProfilePopoutMenu";
import { IconType } from "../ui/icons";
import { MobileNavBar } from "./MobileNavBar";

interface NavLink {
  name: string;
  href: string;
}

interface NavbarProps {
  userName: string;
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
        <ProfilePopoutMenu
          menuItemGroups={menuItemGroups}
          userName={userName}
          userEmail={userEmail}
          avatarSrc={avatarSrc}
        />
      </nav>
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
    </>
  );
};
