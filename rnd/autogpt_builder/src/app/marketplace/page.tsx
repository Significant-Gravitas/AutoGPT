"use client";
import React, { useEffect, useMemo, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import MarketplaceAPI, {
  AgentResponse,
  AgentListResponse,
  AgentWithRank,
} from "@/lib/marketplace-api";
import { ChevronLeft, ChevronRight, Search, Star } from "lucide-react";

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
const HeroSection: React.FC = () => (
  <div className="relative bg-indigo-600 py-6">
    <div className="absolute inset-0 z-0">
      <img
        className="w-full h-full object-cover opacity-20"
        src="https://images.unsplash.com/photo-1562408590-e32931084e23?auto=format&fit=crop&w=2070&q=80"
        alt="Marketplace background"
      />
      <div
        className="absolute inset-0 bg-indigo-600 mix-blend-multiply"
        aria-hidden="true"
      ></div>
    </div>
    <div className="relative max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
      <h1 className="text-2xl font-extrabold tracking-tight text-white sm:text-3xl lg:text-4xl">
        AutoGPT Marketplace
      </h1>
      <p className="mt-2 max-w-3xl text-sm sm:text-base text-indigo-100">
        Discover and share proven AI Agents to supercharge your business.
      </p>
    </div>
  </div>
);

const SearchInput: React.FC<{
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}> = ({ value, onChange }) => (
  <div className="mb-8 relative">
    <Input
      placeholder="Search agents..."
      type="text"
      className="w-full pl-10 pr-4 py-2 rounded-full border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
      value={value}
      onChange={onChange}
    />
    <Search
      className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
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
      className={`flex flex-col justify-between p-6 cursor-pointer hover:bg-gray-50 transition-colors duration-200 rounded-lg border ${featured ? "border-indigo-500 shadow-md" : "border-gray-200"}`}
      onClick={handleClick}
    >
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold text-gray-900 truncate">
            {agent.name}
          </h3>
          {featured && <Star className="text-indigo-500" size={20} />}
        </div>
        <p className="text-sm text-gray-500 line-clamp-2 mb-4">
          {agent.description}
        </p>
        <div className="text-xs text-gray-400 mb-2">
          Categories: {agent.categories.join(", ")}
        </div>
      </div>
      <div className="flex justify-between items-end">
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
    <h2 className="text-2xl font-bold text-gray-900 mb-4">{title}</h2>
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
  <div className="flex justify-between items-center mt-8">
    <Button
      onClick={onPrevPage}
      disabled={page === 1}
      className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
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
      className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
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
    "http://localhost:8001/api/v1/market";
  const api = useMemo(() => new MarketplaceAPI(apiUrl), [apiUrl]);

  const [searchValue, setSearchValue] = useState("");
  const [searchResults, setSearchResults] = useState<Agent[]>([]);
  const [featuredAgents, setFeaturedAgents] = useState<Agent[]>([]);
  const [topAgents, setTopAgents] = useState<Agent[]>([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(false);

  const fetchTopAgents = useCallback(
    async (currentPage: number) => {
      setIsLoading(true);
      try {
        const response = await api.getTopDownloadedAgents(currentPage, 9);
        setTopAgents(response.agents);
        setTotalPages(response.total_pages);
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
      setFeaturedAgents(featured.agents);
    } catch (error) {
      console.error("Error fetching featured agents:", error);
    }
  }, [api]);

  const searchAgents = useCallback(
    async (searchTerm: string) => {
      setIsLoading(true);
      try {
        const response = await api.searchAgents(searchTerm, 1, 30);
        const filteredAgents = response.filter((agent) => agent.rank > 0);
        setSearchResults(filteredAgents);
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
      debouncedSearch(searchValue);
    } else {
      fetchTopAgents(page);
    }
  }, [searchValue, page, debouncedSearch, fetchTopAgents]);

  useEffect(() => {
    fetchFeaturedAgents();
  }, [fetchFeaturedAgents]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchValue(e.target.value);
    setPage(1);
  };

  const handleNextPage = () => {
    if (page < totalPages) {
      setPage(page + 1);
    }
  };

  const handlePrevPage = () => {
    if (page > 1) {
      setPage(page - 1);
    }
  };

  return (
    <div className="bg-gray-50 min-h-screen">
      <HeroSection />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <SearchInput value={searchValue} onChange={handleInputChange} />
        {isLoading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            <p className="mt-2 text-gray-600">Loading agents...</p>
          </div>
        ) : searchValue ? (
          searchResults.length > 0 ? (
            <AgentGrid agents={searchResults} title="Search Results" />
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-600">
                No agents found matching your search criteria.
              </p>
            </div>
          )
        ) : (
          <>
            {featuredAgents.length > 0 && (
              <AgentGrid
                agents={featuredAgents}
                title="Featured Agents"
                featured={true}
              />
            )}
            <AgentGrid agents={topAgents} title="Top Downloaded Agents" />
            <Pagination
              page={page}
              totalPages={totalPages}
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
