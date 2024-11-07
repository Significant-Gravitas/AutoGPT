import * as React from "react";
import { Navbar } from "@/components/agptui/Navbar";
import { Sidebar } from "@/components/agptui/Sidebar";
import { SettingsInputForm } from "@/components/agptui/SettingsInputForm";
import { IconType } from "@/components/ui/icons";

interface SettingsPageProps {
  isLoggedIn: boolean;
  userName: string;
  userEmail: string;
  navLinks: { name: string; href: string }[];
  activeLink: string;
  menuItemGroups: {
    groupName?: string;
    items: {
      icon: IconType;
      text: string;
      href?: string;
      onClick?: () => void;
    }[];
  }[];
  sidebarLinkGroups: {
    links: {
      text: string;
      href: string;
    }[];
  }[];
}

export const SettingsPage: React.FC<SettingsPageProps> = ({
  isLoggedIn,
  userName,
  userEmail,
  navLinks,
  activeLink,
  menuItemGroups,
  sidebarLinkGroups,
}) => {
  return (
    <div className="mx-auto w-screen max-w-[1440px] bg-white lg:w-full">
      <Navbar
        isLoggedIn={isLoggedIn}
        userName={userName}
        userEmail={userEmail}
        links={navLinks}
        activeLink={activeLink}
        menuItemGroups={menuItemGroups}
      />

      <div className="flex min-h-screen flex-col lg:flex-row">
        <Sidebar linkGroups={sidebarLinkGroups} />
        <main className="flex-1 overflow-hidden px-4 pb-8 pt-4 md:px-6 md:pt-6 lg:px-8 lg:pt-8">
          <SettingsInputForm />
        </main>
      </div>
    </div>
  );
};
