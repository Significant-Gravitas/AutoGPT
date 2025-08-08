import { useState } from "react";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";

interface useFeaturedSectionProps {
  featuredAgents: StoreAgent[];
}

export const useFeaturedSection = ({
  featuredAgents,
}: useFeaturedSectionProps) => {
  const [_, setCurrentSlide] = useState(0);

  const handlePrevSlide = () => {
    setCurrentSlide((prev) =>
      prev === 0 ? featuredAgents.length - 1 : prev - 1,
    );
  };

  const handleNextSlide = () => {
    setCurrentSlide((prev) =>
      prev === featuredAgents.length - 1 ? 0 : prev + 1,
    );
  };

  return {
    handleNextSlide,
    handlePrevSlide,
  };
};
