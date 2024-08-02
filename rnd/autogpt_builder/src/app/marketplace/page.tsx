"use client";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from 'next/navigation';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import MarketplaceAPI, { AgentResponse, AgentListResponse } from "@/lib/marketplace-api";

interface AgentRowProps {
    agent: AgentResponse;
}

const AgentRow = ({ agent }: AgentRowProps) => {
    const router = useRouter();

    const handleClick = () => {
        router.push(`/marketplace/${agent.id}`);
    };

    return (
        <li className="flex justify-between gap-x-6 py-5 cursor-pointer hover:bg-gray-50" onClick={handleClick}>
            <div className="flex min-w-0 gap-x-4">
                <img className="h-12 w-12 flex-none rounded-full bg-gray-50" src="https://images.unsplash.com/photo-1562408590-e32931084e23?q=80&w=3270&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="" />
                <div className="min-w-0 flex-auto">
                    <p className="text-sm font-semibold leading-6 text-gray-900">{agent.name}</p>
                    <p className="mt-1 truncate text-xs leading-5 text-gray-500">{agent.description}</p>
                </div>
            </div>
            <div className="flex shrink-0 items-center gap-x-4">
                <div className="hidden sm:flex sm:flex-col sm:items-end">
                    <p className="text-sm leading-6 text-gray-900">{agent.categories.join(', ')}</p>
                    <p className="mt-1 text-xs leading-5 text-gray-500">
                        Last updated <time dateTime={agent.updatedAt}>{new Date(agent.updatedAt).toLocaleDateString()}</time>
                    </p>
                </div>
                <svg className="h-5 w-5 flex-none text-gray-400" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
                </svg>
            </div>
        </li>
    );
}

const Marketplace = () => {
    const apiUrl = process.env.NEXT_PUBLIC_AGPT_MARKETPLACE_URL;
    const api = useMemo(() => new MarketplaceAPI(apiUrl), [apiUrl]);

    const [searchValue, setSearchValue] = useState("");
    const [agents, setAgents] = useState<AgentResponse[]>([]);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [isLoading, setIsLoading] = useState(false);

    const fetchAgents = async (searchTerm: string, currentPage: number) => {
        setIsLoading(true);
        try {
            let response: AgentListResponse;
            if (searchTerm) {
                response = await api.listAgents({ page: currentPage, page_size: 10, keyword: searchTerm });
            } else {
                response = await api.getTopDownloadedAgents(currentPage, 10);
            }
            setAgents(response.agents);
            setTotalPages(response.total_pages);
        } catch (error) {
            console.error("Error fetching agents:", error);
        } finally {
            console.log("Finished fetching agents");
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchAgents(searchValue, page);
    }, [searchValue, page, api]);

    const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
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
        <div className="relative overflow-hidden bg-white">
            <section aria-labelledby="sale-heading" className="relative mx-auto flex max-w-7xl flex-col items-center px-4 pt-32 mb-10 text-center sm:px-6 lg:px-8">
                <div aria-hidden="true" className="absolute inset-0">
                    <div className="absolute inset-0 mx-auto max-w-7xl overflow-hidden xl:px-8">
                        <img src="https://images.unsplash.com/photo-1562408590-e32931084e23?q=80&w=3270&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="" className="w-full object-cover object-center" />
                    </div>
                    <div className="absolute inset-0 bg-white bg-opacity-75"></div>
                    <div className="absolute inset-0 bg-gradient-to-t from-white via-white"></div>
                </div>
                <div className="mx-auto max-w-2xl lg:max-w-none relative z-10">
                    <h2 id="sale-heading" className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl lg:text-6xl">AutoGPT Marketplace</h2>
                    <p className="mx-auto mt-4 max-w-xl text-xl text-gray-600">Discover and Share proven AI Agents and supercharge your business.</p>
                </div>
            </section>

            <section aria-labelledby="testimonial-heading" className="relative justify-center mx-auto max-w-7xl px-4 sm:px-6 lg:py-8">
                <div className="mb-4 flex justify-center">
                    <Input
                        placeholder="Search…"
                        type="text"
                        className="w-3/4"
                        value={searchValue}
                        onChange={handleSearch}
                    />
                </div>

                {isLoading ? (
                    <div className="text-center">Loading...</div>
                ) : (
                    <>
                        <ul role="list" className="divide-y divide-gray-100">
                            {agents.map((agent) => (
                                <AgentRow agent={agent} key={agent.id} />
                            ))}
                        </ul>
                        <div className="flex justify-between mt-4">
                            <Button onClick={handlePrevPage} disabled={page === 1}>Previous</Button>
                            <span>Page {page} of {totalPages}</span>
                            <Button onClick={handleNextPage} disabled={page === totalPages}>Next</Button>
                        </div>
                    </>
                )}
            </section>
        </div>
    );
};

export default Marketplace;