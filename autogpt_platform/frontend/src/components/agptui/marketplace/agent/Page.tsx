import * as React from "react";
import { Navbar } from "../../Navbar";
import { AgentInfo } from "../../AgentInfo";
import { AgentImages } from "../../AgentImages";
import { BecomeACreator } from "../../BecomeACreator";
import { TopAgentsSection } from "../home/TopAgentsSection";
import { Separator } from "../../../ui/separator";
import { IconType } from "../../../ui/icons";
import Link from "next/link";

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

  return (
    <div className="mx-auto w-screen max-w-[1440px]">
      <Navbar
        userName={userName}
        userEmail={userEmail}
        links={navLinks}
        activeLink={activeLink}
        menuItemGroups={menuItemGroups}
      />
      <main className="px-10">
        <div className="mt-10 mb-6">
          <Link href="/marketplace" className="text-2xl font-medium text-[#272727]">
            Marketplace
          </Link>
          <span className="mx-2 text-2xl font-medium text-[#272727]">/</span>
          <span className="text-2xl font-medium text-[#272727]">{agentInfo.name}</span>
        </div>
        <div className="flex flex-col lg:flex-row gap-5">
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
          <AgentImages images={agentImages} />
        </div>
        <Separator className="my-12" />
        <TopAgentsSection
          topAgents={otherAgentsByCreator}
          onCardClick={handleCardClick}
          sectionTitle={`Other agents by ${agentInfo.creator}`}
        />
        <Separator className="my-12" />
        <TopAgentsSection
          topAgents={similarAgents}
          onCardClick={handleCardClick}
          sectionTitle="Similar agents"
        />
        <Separator className="my-12" />
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
