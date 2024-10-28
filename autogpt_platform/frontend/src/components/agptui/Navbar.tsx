import * as React from "react";
import Link from "next/link";
import { ProfilePopoutMenu } from "./ProfilePopoutMenu";
import { IconType, IconLogIn } from "@/components/ui/icons";
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

{
  /* <div className="w-[1408px] h-20 pl-6 pr-3 py-3 bg-white/5 rounded-bl-2xl rounded-br-2xl border border-white/50 backdrop-blur-[26px] justify-between items-center inline-flex">
    <div className="justify-start items-center gap-11 flex">
        <div className="w-[88.87px] h-10 relative" />
        <div className="justify-start items-center gap-6 flex">
            <div className="px-5 py-4 bg-neutral-800 rounded-2xl justify-start items-center gap-3 flex">
                <div className="w-6 h-6 relative" />
                <div className="text-neutral-50 text-xl font-medium font-['Poppins'] leading-7">Marketplace</div>
            </div>
            <div className="px-5 justify-start items-center gap-3 flex">
                <div className="w-6 h-6 relative" />
                <div className="text-neutral-900 text-xl font-medium font-['Poppins'] leading-7">Monitor</div>
            </div>
            <div className="px-5 py-4 rounded-2xl justify-start items-center gap-3 flex">
                <div className="w-6 h-6 relative" />
                <div className="text-neutral-900 text-xl font-medium font-['Poppins'] leading-7">Build</div>
            </div>
        </div>
    </div>
    <div className="justify-start items-center gap-4 flex">
        <div className="p-4 bg-neutral-200 rounded-2xl justify-start items-center gap-2.5 flex">
            <div className="justify-start items-center gap-0.5 flex">
                <div className="text-neutral-900 text-base font-semibold font-['Geist'] leading-7">1500</div>
                <div className="text-neutral-900 text-base font-normal font-['Geist'] leading-7">credits</div>
            </div>
            <div className="w-6 h-6 relative" />
        </div>
        <img className="w-[60px] h-[60px] rounded-full" src="https://via.placeholder.com/60x60" />
    </div>
</div> */
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
      <nav className="sticky top-0 hidden h-20 w-[1408px] items-center justify-between rounded-bl-2xl rounded-br-2xl border border-white/50 bg-white/5 py-3 pl-6 pr-3 backdrop-blur-[26px] md:inline-flex">
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
