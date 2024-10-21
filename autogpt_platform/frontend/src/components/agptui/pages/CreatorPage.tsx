import * as React from "react";
import { Navbar } from "../Navbar";
import { CreatorDetails } from "../composite/CreatorDetails";
import { AgentsSection } from "../composite/AgentsSection";
import { BreadCrumbs } from "../BreadCrumbs";
import { IconType } from "../../ui/icons";

interface CreatorPageProps {
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
        userName={userName}
        userEmail={userEmail}
        links={navLinks}
        activeLink={activeLink}
        menuItemGroups={menuItemGroups}
      />
      <main className="px-10 py-8">
        <BreadCrumbs items={breadcrumbs} />
        <div className="mt-8">
          <CreatorDetails
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
