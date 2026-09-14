import { useRouter } from "next/navigation";
import { useState } from "react";

interface Args {
  defaultValue?: string;
  /** Search a single surface in place; without it the bar goes to the
   *  marketplace-wide results page. */
  onSubmit?: (query: string) => void;
}

export const useSearchbar = ({ defaultValue = "", onSubmit }: Args = {}) => {
  const router = useRouter();

  const [searchQuery, setSearchQuery] = useState(defaultValue);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (onSubmit) {
      onSubmit(searchQuery.trim());
      return;
    }
    if (searchQuery.trim()) {
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
