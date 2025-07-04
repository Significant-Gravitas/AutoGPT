import { Creator } from "@/app/api/__generated__/models/creator";
import { useRouter } from "next/navigation";

interface useFeaturedCreatorsProps {
  featuredCreators: Creator[];
}

export const useFeaturedCreators = ({
  featuredCreators,
}: useFeaturedCreatorsProps) => {
  const router = useRouter();

  const handleCardClick = (creator: string) => {
    router.push(`/marketplace/creator/${encodeURIComponent(creator)}`);
  };
  // Only show first 4 creators
  const displayedCreators = featuredCreators.slice(0, 4);

  return {
    handleCardClick,
    displayedCreators,
  };
};
