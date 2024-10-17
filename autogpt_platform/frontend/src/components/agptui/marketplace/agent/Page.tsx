import * as React from "react";
import { Navbar } from "../../Navbar";
import { AgentInfo } from "../../AgentInfo";
import { AgentImages } from "../../AgentImages";
import { BecomeACreator } from "../../BecomeACreator";
import { AgentsSection } from "../../AgentsSection";
import { Separator } from "../../../ui/separator";
import { IconType } from "../../../ui/icons";
import { BreadCrumbs } from "../../BreadCrumbs";

interface PageProps {
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
    description: string;
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

export const Page: React.FC<PageProps> = ({
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
        userName={userName}
        userEmail={userEmail}
        links={navLinks}
        activeLink={activeLink}
        menuItemGroups={menuItemGroups}
      />
      <main className="px-4 md:mt-4 lg:mt-8">
        <BreadCrumbs items={breadcrumbs} />

        <div className="flex flex-col gap-5 lg:flex-row">
          <div>
            <AgentInfo
              onRunAgent={handleRunAgent}
              name={agentInfo.name}
              creator={agentInfo.creator}
              description={agentInfo.description}
              rating={agentInfo.rating}
              runs={agentInfo.runs}
              categories={agentInfo.categories}
              lastUpdated={agentInfo.lastUpdated}
              version={agentInfo.version}
            />
          </div>
          <AgentImages images={agentImages} />
        </div>
        <Separator className="my-6" />
        <AgentsSection
          agents={otherAgentsByCreator}
          onCardClick={handleCardClick}
          sectionTitle={`Other agents by ${agentInfo.creator}`}
        />
        <Separator className="my-6" />
        <AgentsSection
          agents={similarAgents}
          onCardClick={handleCardClick}
          sectionTitle="Similar agents"
        />
        <Separator className="my-6" />
        <BecomeACreator
          title="Want to contribute?"
          heading="We're always looking for more Creators!"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
          onButtonClick={handleBecomeCreator}
        />
      </main>
    </div>
  );
};
