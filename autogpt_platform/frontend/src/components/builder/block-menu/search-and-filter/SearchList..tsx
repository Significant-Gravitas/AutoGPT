import { searchingData } from "../../testing_data";
import { useEffect, useState } from "react";
import MarketplaceAgentBlock from "../MarketplaceAgentBlock";
import Block from "../Block";
import UGCAgentBlock from "../UGCAgentBlock";
import AiBlock from "./AiBlock";
import IntegrationBlock from "../IntegrationBlock";
import { SearchItem, useBlockMenuContext } from "../block-menu-provider";
import NoSearchResult from "./NoSearchResult";

const SearchList = () => {
  const { searchQuery, searchData, setSearchData } = useBlockMenuContext();
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    // TEMPORARY FETCHING
    const fetchData = async () => {
      setIsLoading(true);
      try {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        setSearchData(searchingData as SearchItem[]);
      } catch (error) {
        console.error("Error fetching search data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [searchQuery, setSearchData]);

  if (isLoading) {
    return (
      <div className="space-y-2.5 px-4">
        <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
          Search results
        </p>
        {Array(6)
          .fill(0)
          .map((_, i) => (
            <Block.Skeleton key={i} />
          ))}
      </div>
    );
  }

  if (searchData.length === 0) {
    return <NoSearchResult />;
  }

  return (
    <div className="space-y-2.5 px-4">
      <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
        Search results
      </p>
      {searchData.map((item: SearchItem, index: number) => {
        switch (item.type) {
          case "marketing_agent":
            return (
              <MarketplaceAgentBlock
                key={index}
                highlightedText={searchQuery}
                title={item.title}
                image_url={item.image_url}
                creator_name={item.creator_name}
                number_of_runs={item.number_of_runs}
              />
            );
          case "block":
            return (
              <Block
                key={index}
                title={item.title}
                highlightedText={searchQuery}
                description={item.description}
              />
            );
          case "integration_block":
            return (
              <IntegrationBlock
                key={index}
                title={item.title}
                highlightedText={searchQuery}
                description={item.description}
                icon_url={item.icon_url}
              />
            );
          case "my_agent":
            return (
              <UGCAgentBlock
                key={index}
                title={item.title}
                highlightedText={searchQuery}
                image_url={item.image_url}
                version={item.version}
                edited_time={item.edited_time}
              />
            );
          case "ai":
            return (
              <AiBlock
                key={index}
                title={item.title}
                description={item.description}
                ai_name={item.ai_name}
              />
            );
          default:
            return null;
        }
      })}
    </div>
  );
};

export default SearchList;
