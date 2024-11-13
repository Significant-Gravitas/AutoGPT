import AutoGPTServerAPIServerSide from "@/lib/autogpt-server-api/clientServer";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { SearchBar } from "@/components/agptui/SearchBar";
import { FeaturedCreators } from "@/components/agptui/composite/FeaturedCreators";
import { Separator } from "@/components/ui/separator";
import { FilterChips } from "@/components/agptui/FilterChips";

export default async function Page({
  params,
  searchParams,
}: {
  params: { lang: string };
  searchParams: { searchTerm?: string };
}) {
  const search_term = searchParams.searchTerm || "";
  const api = new AutoGPTServerAPIServerSide();
  const { agents } = await api.getStoreAgents({ search_query: search_term });
  const { creators } = await api.getStoreCreators({
    search_query: search_term,
  });

  const handleFilterChange = (selectedFilters: string[]) => {
    console.log(selectedFilters);
  };

  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <h2 className="font-['Geist'] text-base font-medium leading-normal">
            Results for:
          </h2>
          <h1 className="font-['Poppins'] text-2xl font-semibold leading-loose">
            {search_term}
          </h1>
        </div>
        <div className="w-32 px-6 py-3.5">
          <SearchBar />
        </div>
      </div>
      <div className="flex justify-between">
        {/* TODO: Add filter chips */}
        {/* <FilterChips badges={["All", "Agents", "Creators"]} onFilterChange={handleFilterChange} /> */}
        <div className="w-32 px-6 py-3.5">Sort By</div>
      </div>
      <AgentsSection agents={agents} sectionTitle="Search Results" />
      <Separator />
      <FeaturedCreators featuredCreators={creators} />
    </div>
  );
}
