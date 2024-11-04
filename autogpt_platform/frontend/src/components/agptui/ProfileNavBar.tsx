import * as React from "react";
import Image from "next/image";
import { ProfilePopoutMenu } from "@/components/agptui/ProfilePopoutMenu";
import CreditsCard from "./CreditsCard";
import { IconType } from "@/components/ui/icons";

interface ProfileNavBarProps {
  userName?: string;
  userEmail?: string;
  credits?: number;
  avatarSrc?: string;
  onRefreshCredits?: () => void;
  menuItemGroups?: Array<{
    groupName?: string;
    items: Array<{
      icon: IconType;
      text: string;
      href?: string;
      onClick?: () => void;
    }>;
  }>;
}

export const ProfileNavBar: React.FC<ProfileNavBarProps> = ({
  userName,
  userEmail,
  credits = 0,
  avatarSrc,
  onRefreshCredits,
  menuItemGroups = [],
}) => {
  return (
    <nav className="sticky top-0 z-50 w-full bg-white">
      <div className="mx-auto flex h-20 w-full items-center justify-between px-6">
        {/* Logo */}
        <div className="flex h-10 w-20 items-center justify-center">
          <a href="https://agpt.co/" aria-label="AutoGPT Home">
            <Image
              src="/AUTOgpt_Logo_dark.png"
              alt="AutoGPT Logo"
              width={100}
              height={40}
              priority
            />
          </a>
        </div>

        {/* Right side content */}
        <div className="flex items-center gap-4">
          <CreditsCard credits={credits} onRefresh={onRefreshCredits} />
          <ProfilePopoutMenu
            userName={userName}
            userEmail={userEmail}
            avatarSrc={avatarSrc}
            menuItemGroups={menuItemGroups}
            hideNavBarUsername={true}
          />
        </div>
      </div>
    </nav>
  );
};
