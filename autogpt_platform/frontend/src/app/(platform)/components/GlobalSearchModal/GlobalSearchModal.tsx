import type { SearchResultItem } from "@/app/api/__generated__/models/searchResultItem";
import { SearchCommandModal } from "@/components/organisms/SearchCommandModal/SearchCommandModal";
import { useGlobalSearch } from "./useGlobalSearch";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelectItem: (item: SearchResultItem) => void;
}

export function GlobalSearchModal({ isOpen, onClose, onSelectItem }: Props) {
  const { query, setQuery, buckets, itemsById, isFetching, isError } =
    useGlobalSearch(isOpen);

  function handleSelect(id: string) {
    const apiItem = itemsById.get(id);
    if (!apiItem) return;
    onSelectItem(apiItem);
    onClose();
  }

  return (
    <SearchCommandModal
      isOpen={isOpen}
      onClose={onClose}
      query={query}
      onQueryChange={setQuery}
      buckets={buckets}
      isLoading={isFetching}
      isError={isError}
      placeholder="Search agents, files, chats..."
      inputAriaLabel="Global search"
      idleEmptyLabel="No recent items"
      searchingEmptyLabel="No results found"
      onSelectItem={(item) => handleSelect(item.id)}
    />
  );
}
