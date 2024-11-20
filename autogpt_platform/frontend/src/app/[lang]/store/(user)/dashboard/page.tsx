"use client";

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

export default function Page({
  sidebarLinkGroups,
  agents,
}: CreatorDashboardPageProps) {
  return (
    <div className="flex">
      <Sidebar linkGroups={sidebarLinkGroups} />

      <main className="flex-1 px-6 py-8 md:px-10">
        {/* Header Section */}
        <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="font-neue text-3xl font-medium leading-9 tracking-tight text-neutral-900">
              Submit a New Agent
            </h1>
            <p className="mt-2 font-neue text-sm text-[#707070]">
              Select from the list of agents you currently have, or upload from
              your local machine.
            </p>
          </div>
          <Button variant="default" size="lg">
            Create New Agent
          </Button>
        </div>

        <Separator className="mb-8" />

        {/* Agents Section */}
        <div>
          <h2 className="mb-4 text-xl font-bold text-neutral-900">
            Your Agents
          </h2>
          <AgentTable agents={agents} />
        </div>
      </main>
    </div>
  );
}
