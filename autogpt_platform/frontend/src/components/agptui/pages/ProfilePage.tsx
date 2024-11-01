import * as React from "react"
import { Sidebar } from "@/components/agptui/Sidebar"
import { ProfileInfoForm } from "@/components/agptui/ProfileInfoForm"
import { IconType } from "@/components/ui/icons"
import { ProfileNavBar } from "@/components/agptui/ProfileNavBar"

interface ProfilePageProps {
  userName?: string;
  userEmail?: string;
  credits?: number;
  displayName?: string;
  handle?: string;
  bio?: string;
  links?: Array<{ id: number; url: string }>;
  categories?: Array<{ id: number; name: string }>;
  menuItemGroups?: Array<{ items: Array<{ icon: IconType; text: string; href?: string; onClick?: () => void }> }>;
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
  ]

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
  ]

  return (
    <div className="min-h-screen w-full bg-white">
      <header className="relative z-50">
        <ProfileNavBar
          userName={userName}
          userEmail={userEmail}
          credits={credits}
          onRefreshCredits={() => console.log('Refresh credits')}
          menuItemGroups={updatedMenuItemGroups}
        />
      </header>

      <div className="flex flex-1">
        <nav 
          className="fixed left-[10px] top-[80px] z-40"
          aria-label="Main navigation"
        >
          <div className="bg-gray-50 rounded-lg">
            <Sidebar linkGroups={sidebarLinkGroups} />
          </div>
        </nav>

        <div className="w-full lg:ml-[200px] flex-1 px-4 md:px-8 py-6">
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
  )
}

export default ProfilePage
