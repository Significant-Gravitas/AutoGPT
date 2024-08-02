"use client";
import { useEffect, useMemo, useState, useCallback } from "react";
import { useRouter } from 'next/navigation';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import MarketplaceAPI, { AgentResponse, AgentListResponse, AgentWithRank } from "@/lib/marketplace-api";
import { ChevronLeft, ChevronRight, Search } from 'lucide-react';

function debounce<T extends (...args: any[]) => any>(func: T, wait: number): (...args: Parameters<T>) => void {
    let timeout: NodeJS.Timeout | null = null;
    return (...args: Parameters<T>) => {
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
}

interface AgentRowProps {
    agent: AgentResponse | AgentWithRank;
}

const AgentRow = ({ agent }: AgentRowProps) => {
    const router = useRouter();

    const handleClick = () => {
        router.push(`/marketplace/${agent.id}`);
    };

    return (
        <li
            className="flex flex-col md:flex-row justify-between gap-4 py-6 px-4 cursor-pointer hover:bg-gray-50 transition-colors duration-200 rounded-lg"
            onClick={handleClick}
        >
            <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center text-2xl font-bold text-gray-500">
                    {agent.name.charAt(0)}
                </div>
                <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold text-gray-900 truncate">{agent.name}</h3>
                    <p className="mt-1 text-sm text-gray-500 line-clamp-2">{agent.description}</p>
                </div>
            </div>
            <div className="flex flex-col items-end justify-center">
                <div className="text-sm text-gray-500">{agent.categories.join(', ')}</div>
                <div className="mt-1 text-xs text-gray-400">
                    Updated {new Date(agent.updatedAt).toLocaleDateString()}
                </div>
                {'rank' in agent && (
                    <div className="mt-1 text-xs text-indigo-600">
                        Rank: {agent.rank.toFixed(2)}
                    </div>
                )}
            </div>
        </li>
    );
}

const Marketplace = () => {
    const apiUrl = process.env.NEXT_PUBLIC_AGPT_MARKETPLACE_URL;
    const api = useMemo(() => new MarketplaceAPI(apiUrl), [apiUrl]);

    const [searchValue, setSearchValue] = useState("");
    const [agents, setAgents] = useState<(AgentResponse | AgentWithRank)[]>([]);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [isLoading, setIsLoading] = useState(false);

    const fetchAgents = useCallback(async (searchTerm: string, currentPage: number) => {
        setIsLoading(true);
        try {
            let response: AgentListResponse | AgentWithRank[];
            if (searchTerm) {
                response = await api.searchAgents(searchTerm, currentPage, 10);
                const filteredAgents = (response as AgentWithRank[]).filter(agent => agent.rank > 0);
                setAgents(filteredAgents);
                setTotalPages(Math.ceil(filteredAgents.length / 10));
            } else {
                response = await api.getTopDownloadedAgents(currentPage, 10);
                setAgents(response.agents);
                setTotalPages(response.total_pages);
            }
        } catch (error) {
            console.error("Error fetching agents:", error);
        } finally {
            setIsLoading(false);
        }
    }, [api]);

    const debouncedFetchAgents = useMemo(
        () => debounce(fetchAgents, 300),
        [fetchAgents]
    );

    useEffect(() => {
        debouncedFetchAgents(searchValue, page);
    }, [searchValue, page, debouncedFetchAgents]);

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSearchValue(e.target.value);
        setPage(1); // Reset to first page on new search
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
            <div className="relative bg-indigo-600 py-24">
                <div className="absolute inset-0">
                    <img
                        className="w-full h-full object-cover opacity-20"
                        src="https://images.unsplash.com/photo-1562408590-e32931084e23?auto=format&fit=crop&w=2070&q=80"
                        alt="Marketplace background"
                    />
                    <div className="absolute inset-0 bg-indigo-600 mix-blend-multiply" aria-hidden="true"></div>
                </div>
                <div className="relative max-w-7xl mx-auto py-24 px-4 sm:px-6 lg:px-8">
                    <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-6xl">AutoGPT Marketplace</h1>
                    <p className="mt-6 max-w-3xl text-xl text-indigo-100">Discover and share proven AI Agents to supercharge your business. Explore our curated collection of powerful tools designed to enhance productivity and innovation.</p>
                </div>
            </div>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="mb-8 relative">
                    <Input
                        placeholder="Search agents..."
                        type="text"
                        className="w-full pl-10 pr-4 py-2 rounded-full border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
                        value={searchValue}
                        onChange={handleInputChange}
                    />
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                </div>

                {isLoading ? (
                    <div className="text-center py-12">
                        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
                        <p className="mt-2 text-gray-600">Loading agents...</p>
                    </div>
                ) : agents.length > 0 ? (
                    <>
                        <h2 className="text-2xl font-bold text-gray-900 mb-4">
                            {searchValue ? "Search Results" : "Top Downloaded Agents"}
                        </h2>
                        <ul className="space-y-4">
                            {agents.map((agent) => (
                                <AgentRow agent={agent} key={agent.id} />
                            ))}
                        </ul>
                        <div className="flex justify-between items-center mt-8">
                            <Button
                                onClick={handlePrevPage}
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
                                onClick={handleNextPage}
                                disabled={page === totalPages}
                                className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                            >
                                <span>Next</span>
                                <ChevronRight size={16} />
                            </Button>
                        </div>
                    </>
                ) : (
                    <div className="text-center py-12">
                        <p className="text-gray-600">No agents found matching your search criteria.</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Marketplace;