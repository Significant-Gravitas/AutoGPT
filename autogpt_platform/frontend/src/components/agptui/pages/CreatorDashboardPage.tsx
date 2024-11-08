import * as React from "react";
import { Navbar } from "@/components/agptui/Navbar";
import { Sidebar } from "@/components/agptui/Sidebar";
import { AgentTable } from "@/components/agptui/AgentTable";
import { Button } from "@/components/agptui/Button";
import { Separator } from "@/components/ui/separator";
import { IconType } from "@/components/ui/icons";

interface CreatorDashboardPageProps {
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
  agents: {
    agentName: string;
    description: string;
    imageSrc: string;
    dateSubmitted: string;
    status: string;
    runs: number;
    rating: number;
    onEdit: () => void;
  }[];
}

export const CreatorDashboardPage: React.FC<CreatorDashboardPageProps> = ({
  isLoggedIn,
  userName,
  userEmail,
  navLinks,
  activeLink,
  menuItemGroups,
  sidebarLinkGroups,
  agents,
}) => {
  return (
    <div className="mx-auto w-full max-w-[1440px] bg-white">
      <Navbar
        isLoggedIn={isLoggedIn}
        userName={userName}
        userEmail={userEmail}
        links={navLinks}
        activeLink={activeLink}
        menuItemGroups={menuItemGroups}
      />

      <div className="flex flex-col md:flex-row">
        <Sidebar linkGroups={sidebarLinkGroups} />

        <main className="flex-1 px-4 py-6 md:px-10 md:py-8">
          {/* Header Section */}
          <div className="mb-6 md:mb-8">
            <h1 className="text-2xl md:text-[32px] font-medium leading-tight md:leading-[38px] text-neutral-900">
              Agent dashboard
            </h1>
          </div>

          {/* Submit Agent Section */}
          <div className="mb-6 md:mb-8">
            <h2 className="mb-1 text-lg md:text-[20px] font-medium leading-tight md:leading-[24px] text-neutral-900">
              Submit an agent
            </h2>
            <div className="flex flex-col md:flex-row md:justify-between md:items-end gap-4 md:gap-0">
              <p className="text-sm md:text-[14px] leading-[20px] text-neutral-600">
                Select from the list of agents you currently already have, or upload from your local machine.
              </p>
              <button className="w-full md:w-auto h-12 px-5 py-3 bg-neutral-800 rounded-[38px] inline-flex items-center justify-center md:justify-start gap-2.5 md:ml-4">
                <span className="text-white text-base font-medium font-['Geist'] leading-normal">Submit agent</span>
              </button>
            </div>
          </div>

          <Separator className="my-6 md:my-8 bg-neutral-300" />

          {/* Agents List Section */}
          <div className="mb-4 md:mb-6">
            <h2 className="text-lg md:text-[20px] font-medium leading-tight md:leading-[24px] text-neutral-900">
              Your uploaded agents
            </h2>
          </div>

          {/* Agent Table */}
          <AgentTable agents={agents} />
        </main>
      </div>
    </div>
  );
};
