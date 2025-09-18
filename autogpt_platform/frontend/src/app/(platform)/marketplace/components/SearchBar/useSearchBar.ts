import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  useTrackEvent,
  EventKeys,
} from "@/services/feature-flags/use-track-event";

export const useSearchbar = () => {
  const router = useRouter();
  const { track } = useTrackEvent();

  const [searchQuery, setSearchQuery] = useState("");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    console.log(searchQuery);

    if (searchQuery.trim()) {
      // Track search performed
      track(EventKeys.STORE_SEARCH_PERFORMED, {
        searchQuery: searchQuery.trim(),
        timestamp: new Date().toISOString(),
      });

      const encodedTerm = encodeURIComponent(searchQuery);
      router.push(`/marketplace/search?searchTerm=${encodedTerm}`);
    }
  };

  return {
    handleSubmit,
    setSearchQuery,
    searchQuery,
  };
};
