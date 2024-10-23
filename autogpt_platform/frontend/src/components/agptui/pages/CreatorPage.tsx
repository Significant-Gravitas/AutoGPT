import * as React from "react";
import { Navbar } from "@/components/agptui/Navbar";
import { CreatorDetails } from "@/components/agptui/composite/CreatorDetails";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { BreadCrumbs } from "@/components/agptui/BreadCrumbs";
import { IconType } from "@/components/ui/icons";

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
      <main className="w-full px-10 py-8">
        <BreadCrumbs items={breadcrumbs} />
        <div className="mt-8">
          <CreatorDetails
            avatarSrc={creatorInfo.avatarSrc}
            name={creatorInfo.name}
            username={creatorInfo.username}
            description={creatorInfo.description}
            avgRating={creatorInfo.avgRating}
            agentCount={creatorInfo.agentCount}
            topCategories={creatorInfo.topCategories}
            otherLinks={creatorInfo.otherLinks}
          />
        </div>
        <div className="mt-16">
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
