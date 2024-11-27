import AutoGPTServerAPIServerSide from "@/lib/autogpt-server-api/clientServer";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { SearchBar } from "@/components/agptui/SearchBar";
import { FeaturedCreators } from "@/components/agptui/composite/FeaturedCreators";
import { Separator } from "@/components/ui/separator";
import { SearchFilterChips } from "@/components/agptui/SearchFilterChips";
import { SortDropdown } from "@/components/agptui/SortDropdown";

export default async function Page({
  params,
  searchParams,
}: {
  params: { lang: string };
  searchParams: { searchTerm?: string; sort?: string };
}) {
  const search_term = searchParams.searchTerm || "";
  const sort = searchParams.sort || "trending";
  
  const api = new AutoGPTServerAPIServerSide();
  const { agents } = await api.getStoreAgents({ 
    search_query: search_term,
    sort: sort 
  });
  const { creators } = await api.getStoreCreators({
    search_query: search_term,
  });

  const agentsCount = agents?.length || 0;
  const creatorsCount = creators?.length || 0;
  const totalCount = agentsCount + creatorsCount;

  return (
    <div className="w-full bg-white">
      <div className="px-10 max-w-[1440px] mx-auto">
        <div className="flex items-center justify-between mt-8">
          <div className="flex flex-col">
            <h2 className="font-['Geist'] text-base font-medium text-neutral-800">
              Results for:
            </h2>
            <h1 className="font-['Poppins'] text-2xl font-semibold text-neutral-800">
              {search_term}
            </h1>
          </div>
          <div>
            <SearchBar width="w-[439px]" />
          </div>
        </div>

        <div className="mt-8 flex justify-between items-center">
          <SearchFilterChips 
            totalCount={totalCount}
            agentsCount={agentsCount}
            creatorsCount={creatorsCount}
          />
          <SortDropdown />
        </div>

        <div className="mt-6">
          <h2 className="text-neutral-800 text-lg font-semibold font-['Poppins'] mb-4">Agents</h2>
          <AgentsSection agents={agents} sectionTitle="Search Results" />
        </div>
        
        <Separator className="my-6" />
        
        <div className="mb-8">
          <h2 className="text-neutral-800 text-lg font-semibold font-['Poppins'] mb-4">Creators</h2>
          <FeaturedCreators featuredCreators={creators} />
        </div>
      </div>
    </div>
  );
}
