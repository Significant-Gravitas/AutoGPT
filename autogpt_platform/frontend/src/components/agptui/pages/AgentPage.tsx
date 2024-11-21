import * as React from "react";
import { Navbar } from "@/components/agptui/Navbar";
import { AgentInfo } from "@/components/agptui/AgentInfo";
import { AgentImages } from "@/components/agptui/AgentImages";
import { BecomeACreator } from "@/components/agptui/BecomeACreator";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { Separator } from "@/components/ui/separator";
import { IconType } from "@/components/ui/icons";
import { BreadCrumbs } from "@/components/agptui/BreadCrumbs";

interface AgentPageProps {
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
  agentInfo: {
    name: string;
    creator: string;
    shortDescription: string;
    longDescription: string;
    rating: number;
    runs: number;
    categories: string[];
    lastUpdated: string;
    version: string;
  };
  agentImages: string[];
  otherAgentsByCreator: {
    agentName: string;
    agentImage: string;
    description: string;
    runs: number;
    rating: number;
    avatarSrc: string;
  }[];
  similarAgents: {
    agentName: string;
    agentImage: string;
    description: string;
    runs: number;
    rating: number;
    avatarSrc: string;
  }[];
}

export const AgentPage: React.FC<AgentPageProps> = ({
  isLoggedIn,
  userName,
  userEmail,
  navLinks,
  activeLink,
  menuItemGroups,
  agentInfo,
  agentImages,
  otherAgentsByCreator,
  similarAgents,
}) => {
  const handleRunAgent = () => {
    console.log("Run agent clicked");
    // Implement run agent functionality
  };

  const handleCardClick = (agentName: string) => {
    console.log("Clicked on agent:", agentName);
    // Implement card click functionality
  };

  const handleBecomeCreator = () => {
    console.log("Become a Creator clicked");
    // Implement become a creator functionality
  };

  const breadcrumbs = [
    { name: "Marketplace", link: "/marketplace" },
    { name: agentInfo.name, link: "#" },
  ];

  return (
    <div className="mx-auto w-screen max-w-[1360px]">
      <Navbar
        isLoggedIn={isLoggedIn}
        userName={userName}
        userEmail={userEmail}
        links={navLinks}
        activeLink={activeLink}
        menuItemGroups={menuItemGroups}
      />
      <main className="px-4 md:mt-4 lg:mt-8">
        <BreadCrumbs items={breadcrumbs} />

        <div className="flex flex-col lg:flex-row lg:gap-8 xl:gap-12">
          <div className="w-full lg:max-w-[396px]">
            <AgentInfo
              name={agentInfo.name}
              creator={agentInfo.creator}
              shortDescription={agentInfo.shortDescription}
              longDescription={agentInfo.longDescription}
              rating={agentInfo.rating}
              runs={agentInfo.runs}
              categories={agentInfo.categories}
              lastUpdated={agentInfo.lastUpdated}
              version={agentInfo.version}
            />
          </div>
          <div className="flex-1">
            <AgentImages images={agentImages} />
          </div>
        </div>

        <Separator className="my-6" />
        <AgentsSection
          agents={otherAgentsByCreator.map((agent) => ({
            slug: agent.agentName.toLowerCase().replace(/\s+/g, "-"),
            agent_name: agent.agentName,
            agent_image: agent.agentImage,
            creator: agentInfo.creator,
            creator_avatar: agent.avatarSrc,
            sub_heading: "",
            description: agent.description,
            runs: agent.runs,
            rating: agent.rating,
          }))}
          onCardClick={handleCardClick}
          sectionTitle={`Other agents by ${agentInfo.creator}`}
        />
        <Separator className="my-6" />
        <AgentsSection
          agents={similarAgents.map((agent) => ({
            slug: agent.agentName.toLowerCase().replace(/\s+/g, "-"),
            agent_name: agent.agentName,
            agent_image: agent.agentImage,
            creator: agentInfo.creator,
            creator_avatar: agent.avatarSrc,
            sub_heading: "",
            description: agent.description,
            runs: agent.runs,
            rating: agent.rating,
          }))}
          onCardClick={handleCardClick}
          sectionTitle="Similar agents"
        />
        <BecomeACreator
          title="Want to contribute?"
          heading="We're always looking for more Creators!"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
};
