import * as React from "react";
import { HeroSection } from "@/components/agptui/composite/HeroSection";
import { FeaturedSection } from "@/components/agptui/composite/FeaturedSection";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { BecomeACreator } from "@/components/agptui/BecomeACreator";
import { FeaturedCreators } from "@/components/agptui/composite/FeaturedCreators";
import { Separator } from "@/components/ui/separator";
import { getFeaturedAgents } from "@/app/api/featuredAgents/getFeaturedAgents";

interface PageProps {
  featuredAgents: {
    agentName: string;
    agentImage: string;
    creatorName: string;
    description: string;
    runs: number;
    rating: number;
  }[];
  topAgents: {
    agentName: string;
    agentImage: string;
    avatarSrc: string;
    description: string;
    runs: number;
    rating: number;
  }[];
  featuredCreators: {
    creatorName: string;
    creatorImage: string;
    bio: string;
    agentsUploaded: number;
    avatarSrc: string;
  }[];
}

export default async function Page() {
  let featuredAgents = await getFeaturedAgents();

  let topAgents = [
    {
      agentName: "Data Analyzer Pro",
      agentImage:
        "https://ddz4ak4pa3d19.cloudfront.net/cache/07/78/0778415062f8dff56a046a7eca44567c.jpg",
      avatarSrc: "https://github.com/shadcn.png",
      description:
        "Powerful tool for analyzing large datasets and generating insights.",
      runs: 50000,
      rating: 5,
    },
    {
      agentName: "Image Recognition Master",
      agentImage:
        "https://ddz4ak4pa3d19.cloudfront.net/cache/59/b9/59b9415d4044f48f9b9e318c4c5a7984.jpg",
      avatarSrc: "https://example.com/avatar2.jpg",
      description:
        "Accurately identify and classify objects in images using state-of-the-art machine learning algorithms.",
      runs: 60000,
      rating: 4.6,
    },
  ];
  let featuredCreators = [
    {
      creatorName: "AI Labs",
      creatorImage:
        "https://ddz4ak4pa3d19.cloudfront.net/cache/53/b2/53b2bc7d7900f0e1e60bf64ebf38032d.jpg",
      bio: "Pioneering AI solutions for everyday problems",
      agentsUploaded: 25,
      avatarSrc: "https://github.com/shadcn.png",
    },
    {
      creatorName: "WriteRight Inc.",
      creatorImage:
        "https://ddz4ak4pa3d19.cloudfront.net/cache/40/f7/40f7bc97c952f8df0f9c88d29defe8d4.jpg",
      bio: "Empowering content creators with AI-driven tools",
      agentsUploaded: 18,
      avatarSrc: "https://example.com/writeright-avatar.jpg",
    },
  ];

  const handleSearch = (query: string) => {
    console.log("Search query:", query);
    // Implement search functionality
  };

  const handleFilterChange = (selectedFilters: string[]) => {
    console.log("Selected filters:", selectedFilters);
    // Implement filter functionality
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
    <div className="mx-auto w-screen max-w-[1360px]">
      <main className="px-4">
        <HeroSection
          onSearch={handleSearch}
          onFilterChange={handleFilterChange}
        />
        <FeaturedSection
          featuredAgents={featuredAgents}
          onCardClick={handleCardClick}
        />
        <Separator />
        <AgentsSection
          sectionTitle="Top Agents"
          agents={topAgents}
          onCardClick={handleCardClick}
        />
        <Separator />
        <FeaturedCreators
          featuredCreators={featuredCreators}
          onCardClick={handleCardClick}
        />
        <Separator />
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
}
