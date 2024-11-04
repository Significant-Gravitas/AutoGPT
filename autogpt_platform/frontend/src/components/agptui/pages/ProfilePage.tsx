import * as React from "react";
import { Sidebar } from "@/components/agptui/Sidebar";
import { ProfileInfoForm } from "@/components/agptui/ProfileInfoForm";
import { IconType } from "@/components/ui/icons";
import { ProfileNavBar } from "@/components/agptui/ProfileNavBar";

interface ProfilePageProps {
  userName?: string;
  userEmail?: string;
  credits?: number;
  displayName?: string;
  handle?: string;
  bio?: string;
  links?: Array<{ id: number; url: string }>;
  categories?: Array<{ id: number; name: string }>;
  menuItemGroups?: Array<{
    items: Array<{
      icon: IconType;
      text: string;
      href?: string;
      onClick?: () => void;
    }>;
  }>;
}

const ProfilePage = ({
  userName = "",
  userEmail = "",
  credits = 0,
  displayName = "",
  handle = "",
  bio = "",
  links = [],
  categories = [],
  menuItemGroups = [],
}: ProfilePageProps) => {
  const sidebarLinkGroups = [
    {
      links: [
        { text: "Creator Dashboard", href: "/dashboard" },
        { text: "Agent dashboard", href: "/agent-dashboard" },
        { text: "Integrations", href: "/integrations" },
        { text: "Profile", href: "/profile" },
        { text: "Settings", href: "/settings" },
      ],
    },
  ];

  const updatedMenuItemGroups = [
    {
      items: [
        { icon: IconType.Edit, text: "Edit profile", href: "/profile/edit" },
      ],
    },
    {
      items: [
        {
          icon: IconType.LayoutDashboard,
          text: "Creator Dashboard",
          href: "/dashboard",
        },
        {
          icon: IconType.UploadCloud,
          text: "Publish an agent",
          href: "/publish",
        },
      ],
    },
    {
      items: [{ icon: IconType.Settings, text: "Settings", href: "/settings" }],
    },
    {
      items: [
        {
          icon: IconType.LogOut,
          text: "Log out",
          onClick: () => console.log("Logged out"),
        },
      ],
    },
  ];

  return (
    <div className="min-h-screen w-full bg-white">
      <header className="fixed left-0 right-0 top-0 z-50 bg-white">
        <ProfileNavBar
          userName={userName}
          userEmail={userEmail}
          credits={credits}
          onRefreshCredits={() => console.log("Refresh credits")}
          menuItemGroups={updatedMenuItemGroups}
        />
      </header>

      <div className="pt-[64px]">
        <div className="flex flex-1">
          <nav
            className="fixed left-[10px] top-[80px] z-40"
            aria-label="Main navigation"
          >
            <div className="rounded-lg bg-gray-50">
              <Sidebar linkGroups={sidebarLinkGroups} />
            </div>
          </nav>

          <div className="w-full flex-1 px-4 py-6 md:px-8 lg:ml-[200px]">
            <ProfileInfoForm
              displayName={displayName}
              handle={handle}
              bio={bio}
              links={links}
              categories={categories}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;
