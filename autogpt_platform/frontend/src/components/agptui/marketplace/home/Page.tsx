import * as React from "react";
import { Navbar } from "../../Navbar";
import { HeroSection } from "./HeroSection";
import { FeaturedSection } from "./FeaturedSection";
import { TopAgentsSection } from "./TopAgentsSection";
import { BecomeACreator } from "../../BecomeACreator";
import { FeaturedCreators } from "./FeaturedCreators";

interface PageProps {
  userName: string;
  navLinks: { name: string; href: string }[];
  activeLink: string;
  featuredAgents: {
    agentName: string;
    creatorName: string;
    description: string;
    runs: number;
    rating: number;
  }[];
  topAgents: {
    agentName: string;
    description: string;
    runs: number;
    rating: number;
  }[];
  featuredCreators: {
    creatorName: string;
    bio: string;
    agentsUploaded: number;
    avatarSrc: string;
  }[];
}

export const Page: React.FC<PageProps> = ({
  userName,
  navLinks,
  activeLink,
  featuredAgents,
  topAgents,
  featuredCreators,
}) => {
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

  const handleProfileClick = () => {
    console.log("Profile clicked");
    // Implement profile click functionality
  };

  const handleBecomeCreator = () => {
    console.log("Become a Creator clicked");
    // Implement become a creator functionality
  };

  return (
    <div className="mx-auto flex w-screen max-w-[1360px] flex-col items-center">
      <Navbar
        userName={userName}
        links={navLinks}
        activeLink={activeLink}
        onProfileClick={handleProfileClick}
      />
      <main className="mt-8 flex max-w-[1360px] flex-col items-center pb-32">
        <HeroSection
          onSearch={handleSearch}
          onFilterChange={handleFilterChange}
        />
        <FeaturedSection
          featuredAgents={featuredAgents}
          onCardClick={handleCardClick}
        />
        <TopAgentsSection topAgents={topAgents} onCardClick={handleCardClick} />
        <FeaturedCreators
          featuredCreators={featuredCreators}
          onCardClick={handleCardClick}
        />
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
