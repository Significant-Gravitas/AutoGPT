import { IconAutoGPTLogo, IconType } from "@/components/ui/icons";
import Wallet from "../../agptui/Wallet";
import { AccountMenu } from "./components/AccountMenu/AccountMenu";
import { LoginButton } from "./components/LoginButton";
import { MobileNavBar } from "./components/MobileNavbar/MobileNavBar";
import { NavbarLink } from "./components/NavbarLink";
import { accountMenuItems, loggedInLinks, loggedOutLinks } from "./helpers";
import { getNavbarAccountData } from "./data";

export async function Navbar() {
  const { profile, isLoggedIn } = await getNavbarAccountData();

  return (
    <>
      <nav className="sticky top-0 z-40 hidden h-16 items-center rounded-bl-2xl rounded-br-2xl border border-white/50 bg-[#f3f4f6]/20 p-3 backdrop-blur-[26px] md:inline-flex">
        {/* Left section */}
        <div className="flex flex-1 items-center gap-5">
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
                avatarSrc={profile?.avatar_url ?? ""}
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
              avatarSrc={profile?.avatar_url ?? ""}
            />
          </div>
        ) : null}
      </>
    </>
  );
}
