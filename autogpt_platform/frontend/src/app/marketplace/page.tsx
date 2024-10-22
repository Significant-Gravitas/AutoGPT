"use client";
import React, { useEffect, useMemo, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import MarketplaceAPI, {
  AgentResponse,
  AgentWithRank,
} from "@/lib/marketplace-api";
import {
  ChevronLeft,
  ChevronRight,
  PlusCircle,
  Search,
  Star,
} from "lucide-react";

// Utility Functions
function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

// Types
type Agent = AgentResponse | AgentWithRank;

// Components
const HeroSection: React.FC = () => {
  const router = useRouter();

  return (
    <div className="relative bg-indigo-600 py-6">
      <div className="absolute inset-0 z-0">
        <Image
          src="https://images.unsplash.com/photo-1562408590-e32931084e23?auto=format&fit=crop&w=2070&q=80"
          alt="Marketplace background"
          layout="fill"
          objectFit="cover"
          quality={75}
          priority
          className="opacity-20"
        />
        <div
          className="absolute inset-0 bg-indigo-600 mix-blend-multiply"
          aria-hidden="true"
        ></div>
      </div>
      <div className="relative mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-white sm:text-3xl lg:text-4xl">
            AutoGPT Marketplace
          </h1>
          <p className="mt-2 max-w-3xl text-sm text-indigo-100 sm:text-base">
            Discover and share proven AI Agents to supercharge your business.
          </p>
        </div>
        <Button
          onClick={() => router.push("/marketplace/submit")}
          className="flex items-center bg-white text-indigo-600 hover:bg-indigo-50"
        >
          <PlusCircle className="mr-2 h-4 w-4" />
          Submit Agent
        </Button>
      </div>
    </div>
  );
};

const SearchInput: React.FC<{
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}> = ({ value, onChange }) => (
  <div className="relative mb-8">
    <Input
      placeholder="Search agents..."
      type="text"
      className="w-full rounded-full border-gray-300 py-2 pl-10 pr-4 focus:border-indigo-500 focus:ring-indigo-500"
      value={value}
      onChange={onChange}
    />
    <Search
      className="absolute left-3 top-1/2 -translate-y-1/2 transform text-gray-400"
      size={20}
    />
  </div>
);

const AgentCard: React.FC<{ agent: Agent; featured?: boolean }> = ({
  agent,
  featured = false,
}) => {
  const router = useRouter();

  const handleClick = () => {
    router.push(`/marketplace/${agent.id}`);
  };

  return (
    <div
      className={`flex cursor-pointer flex-col justify-between rounded-lg border p-6 transition-colors duration-200 hover:bg-gray-50 ${featured ? "border-indigo-500 shadow-md" : "border-gray-300"}`}
      onClick={handleClick}
    >
      <div>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="truncate text-lg font-semibold text-gray-900">
            {agent.name}
          </h3>
          {featured && <Star className="text-indigo-500" size={20} />}
        </div>
        <p className="mb-4 line-clamp-2 text-sm text-gray-500">
          {agent.description}
        </p>
        <div className="mb-2 text-xs text-gray-400">
          Categories: {agent.categories?.join(", ")}
        </div>
      </div>
      <div className="flex items-end justify-between">
        <div className="text-xs text-gray-400">
          Updated {new Date(agent.updatedAt).toLocaleDateString()}
        </div>
        <div className="text-xs text-gray-400">Downloads {agent.downloads}</div>
        {"rank" in agent && (
          <div className="text-xs text-indigo-600">
            Rank: {agent.rank.toFixed(2)}
          </div>
        )}
      </div>
    </div>
  );
};

const AgentGrid: React.FC<{
  agents: Agent[];
  title: string;
  featured?: boolean;
}> = ({ agents, title, featured = false }) => (
  <div className="mb-12">
    <h2 className="mb-4 text-2xl font-bold text-gray-900">{title}</h2>
    <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
      {agents.map((agent) => (
        <AgentCard agent={agent} key={agent.id} featured={featured} />
      ))}
    </div>
  </div>
);

const Pagination: React.FC<{
  page: number;
  totalPages: number;
  onPrevPage: () => void;
  onNextPage: () => void;
}> = ({ page, totalPages, onPrevPage, onNextPage }) => (
  <div className="mt-8 flex items-center justify-between">
    <Button
      onClick={onPrevPage}
      disabled={page === 1}
      className="flex items-center space-x-2 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50"
    >
      <ChevronLeft size={16} />
      <span>Previous</span>
    </Button>
    <span className="text-sm text-gray-700">
      Page {page} of {totalPages}
    </span>
    <Button
      onClick={onNextPage}
      disabled={page === totalPages}
      className="flex items-center space-x-2 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50"
    >
      <span>Next</span>
      <ChevronRight size={16} />
    </Button>
  </div>
);

// Main Component
const Marketplace: React.FC = () => {
  const apiUrl =
    process.env.NEXT_PUBLIC_AGPT_MARKETPLACE_URL ||
    "http://localhost:8015/api/v1/market";
  const api = useMemo(() => new MarketplaceAPI(apiUrl), [apiUrl]);

  const [searchValue, setSearchValue] = useState("");
  const [searchResults, setSearchResults] = useState<Agent[]>([]);
  const [featuredAgents, setFeaturedAgents] = useState<Agent[]>([]);
  const [topAgents, setTopAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [topAgentsPage, setTopAgentsPage] = useState(1);
  const [searchPage, setSearchPage] = useState(1);
  const [topAgentsTotalPages, setTopAgentsTotalPages] = useState(1);
  const [searchTotalPages, setSearchTotalPages] = useState(1);

  const fetchTopAgents = useCallback(
    async (currentPage: number) => {
      setIsLoading(true);
      try {
        const response = await api.getTopDownloadedAgents(currentPage, 9);
        setTopAgents(response.items);
        setTopAgentsTotalPages(response.total_pages);
      } catch (error) {
        console.error("Error fetching top agents:", error);
      } finally {
        setIsLoading(false);
      }
    },
    [api],
  );

  const fetchFeaturedAgents = useCallback(async () => {
    try {
      const featured = await api.getFeaturedAgents();
      setFeaturedAgents(featured.items);
    } catch (error) {
      console.error("Error fetching featured agents:", error);
    }
  }, [api]);

  const searchAgents = useCallback(
    async (searchTerm: string, currentPage: number) => {
      setIsLoading(true);
      try {
        const response = await api.searchAgents(searchTerm, currentPage, 9);
        const filteredAgents = response.items.filter((agent) => agent.rank > 0);
        setSearchResults(filteredAgents);
        setSearchTotalPages(response.total_pages);
      } catch (error) {
        console.error("Error searching agents:", error);
      } finally {
        setIsLoading(false);
      }
    },
    [api],
  );

  const debouncedSearch = useMemo(
    () => debounce(searchAgents, 300),
    [searchAgents],
  );

  useEffect(() => {
    if (searchValue) {
      searchAgents(searchValue, searchPage);
    } else {
      fetchTopAgents(topAgentsPage);
    }
  }, [searchValue, searchPage, topAgentsPage, searchAgents, fetchTopAgents]);

  useEffect(() => {
    fetchFeaturedAgents();
  }, [fetchFeaturedAgents]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchValue(e.target.value);
    setSearchPage(1);
  };

  const handleNextPage = () => {
    if (searchValue) {
      if (searchPage < searchTotalPages) {
        setSearchPage(searchPage + 1);
      }
    } else {
      if (topAgentsPage < topAgentsTotalPages) {
        setTopAgentsPage(topAgentsPage + 1);
      }
    }
  };

  const handlePrevPage = () => {
    if (searchValue) {
      if (searchPage > 1) {
        setSearchPage(searchPage - 1);
      }
    } else {
      if (topAgentsPage > 1) {
        setTopAgentsPage(topAgentsPage - 1);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <HeroSection />
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <SearchInput value={searchValue} onChange={handleInputChange} />
        {isLoading ? (
          <div className="py-12 text-center">
            <div className="inline-block h-8 w-8 animate-spin rounded-full border-b-2 border-gray-900"></div>
            <p className="mt-2 text-gray-600">Loading agents...</p>
          </div>
        ) : searchValue ? (
          searchResults.length > 0 ? (
            <>
              <AgentGrid agents={searchResults} title="Search Results" />
              <Pagination
                page={searchPage}
                totalPages={searchTotalPages}
                onPrevPage={handlePrevPage}
                onNextPage={handleNextPage}
              />
            </>
          ) : (
            <div className="py-12 text-center">
              <p className="text-gray-600">
                No agents found matching your search criteria.
              </p>
            </div>
          )
        ) : (
          <>
            {featuredAgents?.length > 0 ? (
              <AgentGrid
                agents={featuredAgents}
                title="Featured Agents"
                featured={true}
              />
            ) : (
              <div className="py-12 text-center">
                <p className="text-gray-600">No Featured Agents found</p>
              </div>
            )}

            <hr />

            {topAgents?.length > 0 ? (
              <AgentGrid agents={topAgents} title="Top Downloaded Agents" />
            ) : (
              <div className="py-12 text-center">
                <p className="text-gray-600">No Top Downloaded Agents found</p>
              </div>
            )}
            <Pagination
              page={topAgentsPage}
              totalPages={topAgentsTotalPages}
              onPrevPage={handlePrevPage}
              onNextPage={handleNextPage}
            />
          </>
        )}
      </div>
    </div>
  );
};

export default Marketplace;
