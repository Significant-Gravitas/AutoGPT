import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { SearchResponse } from "@/app/api/__generated__/models/searchResponse";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";

export const getSearchItemType = (
  item: SearchResponse["items"][number],
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

/**
 * A React key per search result that stays with the item when the list
 * changes. Results of different types can share an id, and a later page can
 * repeat an earlier item, so the type is part of the key and a repeat falls
 * back to its position.
 */
export function getSearchItemKeys(items: SearchResponse["items"][number][]) {
  const seen = new Set<string>();
  return items.map((item, index) => {
    const { type, data } = getSearchItemType(item);
    const id = type === "store_agent" ? data.slug : data.id;
    const key = `${type}-${id}`;
    if (!id || seen.has(key)) return `${key}-${index}`;
    seen.add(key);
    return key;
  });
}
