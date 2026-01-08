import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { SearchResponseItemsItem } from "@/app/api/__generated__/models/searchResponseItemsItem";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";

export const getSearchItemType = (
  item: SearchResponseItemsItem,
):
  | { type: "store_agent"; data: StoreAgent }
  | { type: "library_agent"; data: LibraryAgent }
  | { type: "block"; data: BlockInfo } => {
  if ("slug" in item && "agent_name" in item && "creator" in item) {
    return { type: "store_agent", data: item as StoreAgent };
  }

  if ("graph_id" in item && "graph_version" in item && "creator_name" in item) {
    return { type: "library_agent", data: item as LibraryAgent };
  }

  if ("inputSchema" in item && "outputSchema" in item && "uiType" in item) {
    return { type: "block", data: item as BlockInfo };
  }

  throw new Error("Unknown item type");
};
