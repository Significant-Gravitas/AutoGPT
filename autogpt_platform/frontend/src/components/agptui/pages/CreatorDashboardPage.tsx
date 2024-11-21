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
            <h1 className="text-2xl font-medium leading-tight text-neutral-900 md:text-[32px] md:leading-[38px]">
              Agent dashboard
            </h1>
          </div>

          {/* Submit Agent Section */}
          <div className="mb-6 md:mb-8">
            <h2 className="mb-1 text-lg font-medium leading-tight text-neutral-900 md:text-[20px] md:leading-[24px]">
              Submit an agent
            </h2>
            <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between md:gap-0">
              <p className="text-sm leading-[20px] text-neutral-600 md:text-[14px]">
                Select from the list of agents you currently already have, or
                upload from your local machine.
              </p>
              <button className="inline-flex h-12 w-full items-center justify-center gap-2.5 rounded-[38px] bg-neutral-800 px-5 py-3 md:ml-4 md:w-auto md:justify-start">
                <span className="font-['Geist'] text-base font-medium leading-normal text-white">
                  Submit agent
                </span>
              </button>
            </div>
          </div>

          <Separator className="my-6 bg-neutral-300 md:my-8" />

          {/* Agents List Section */}
          <div className="mb-4 md:mb-6">
            <h2 className="text-lg font-medium leading-tight text-neutral-900 md:text-[20px] md:leading-[24px]">
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
