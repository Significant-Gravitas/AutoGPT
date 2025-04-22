import { ReactNode } from "react";
import { Navbar } from "@/components/agptui/Navbar";
import { IconType } from "@/components/ui/icons";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <Navbar
        links={[
          {
            name: "Marketplace",
            href: "/marketplace",
          },
          {
            name: "Library",
            href: "/library",
          },
          {
            name: "Build",
            href: "/build",
          },
        ]}
        menuItemGroups={[
          {
            items: [
              {
                icon: IconType.Edit,
                text: "Edit profile",
                href: "/profile",
              },
            ],
          },
          {
            items: [
              {
                icon: IconType.LayoutDashboard,
                text: "Creator Dashboard",
                href: "/profile/dashboard",
              },
              {
                icon: IconType.UploadCloud,
                text: "Publish an agent",
              },
            ],
          },
          {
            items: [
              {
                icon: IconType.Settings,
                text: "Settings",
                href: "/profile/settings",
              },
            ],
          },
          {
            items: [
              {
                icon: IconType.LogOut,
                text: "Log out",
              },
            ],
          },
        ]}
      />
      <main>{children}</main>
    </>
  );
}
