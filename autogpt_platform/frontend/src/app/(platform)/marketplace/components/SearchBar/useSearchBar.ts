import { useRouter } from "next/navigation";
import { useState } from "react";

export const useSearchbar = () => {
  const router = useRouter();

  const [searchQuery, setSearchQuery] = useState("");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    console.log(searchQuery);

    if (searchQuery.trim()) {
      // Encode the search term and navigate to the desired path
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
