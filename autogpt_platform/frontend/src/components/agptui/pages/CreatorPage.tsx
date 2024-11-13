import * as React from "react";
import { Navbar } from "@/components/agptui/Navbar";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { BreadCrumbs } from "@/components/agptui/BreadCrumbs";
import { IconType } from "@/components/ui/icons";
import { CreatorInfoCard } from "@/components/agptui/CreatorInfoCard";
import { CreatorLinks } from "@/components/agptui/CreatorLinks";

interface CreatorPageProps {
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
  creatorInfo: {
    name: string;
    avatarSrc: string;
    username: string;
    description: string;
    avgRating: number;
    agentCount: number;
    topCategories: string[];
    otherLinks: {
      website?: string;
      github?: string;
      linkedin?: string;
    };
  };
  creatorAgents: {
    agentName: string;
    agentImage: string;
    description: string;
    runs: number;
    rating: number;
    avatarSrc: string;
  }[];
}

export const CreatorPage: React.FC<CreatorPageProps> = ({
  isLoggedIn,
  userName,
  userEmail,
  navLinks,
  activeLink,
  menuItemGroups,
  creatorInfo,
  creatorAgents,
}) => {
  const handleCardClick = (agentName: string) => {
    console.log("Clicked on agent:", agentName);
    // Implement card click functionality
  };

  const breadcrumbs = [
    { name: "Marketplace", link: "/marketplace" },
    { name: creatorInfo.name, link: "#" },
  ];

  return (
    <div className="mx-auto w-screen max-w-[1440px]">
      <Navbar
        isLoggedIn={isLoggedIn}
        userName={userName}
        userEmail={userEmail}
        links={navLinks}
        activeLink={activeLink}
        menuItemGroups={menuItemGroups}
      />
      <main className="w-full px-4 py-4 sm:px-6 sm:py-6 md:px-10 md:py-8">
        <BreadCrumbs items={breadcrumbs} />

        <div className="mt-4 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 md:mt-8 md:flex-row md:gap-8">
          <div className="w-full md:w-auto md:shrink-0">
            <CreatorInfoCard
              username={creatorInfo.name}
              handle={creatorInfo.username}
              avatarSrc={creatorInfo.avatarSrc}
              categories={creatorInfo.topCategories}
              averageRating={creatorInfo.avgRating}
              totalRuns={creatorInfo.agentCount}
            />
          </div>
          <div className="flex min-w-0 flex-1 flex-col gap-4 sm:gap-6 md:gap-8">
            <div className="font-neue text-2xl font-normal leading-normal text-neutral-900 sm:text-3xl md:text-[35px] md:leading-[45px]">
              {creatorInfo.description}
            </div>
            <CreatorLinks links={creatorInfo.otherLinks} />
          </div>
        </div>
        <div className="mt-8 sm:mt-12 md:mt-16">
          <hr className="w-full bg-neutral-700" />
          <AgentsSection
            agents={creatorAgents}
            hideAvatars={true}
            onCardClick={handleCardClick}
            sectionTitle={`Agents by ${creatorInfo.name}`}
          />
        </div>
      </main>
    </div>
  );
};
