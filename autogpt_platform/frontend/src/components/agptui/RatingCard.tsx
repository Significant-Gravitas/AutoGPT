"use client";

import * as React from "react";
import { Cross1Icon } from "@radix-ui/react-icons";
import { IconStar, IconStarFilled } from "@/components/ui/icons";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

interface RatingCardProps {
  agentName: string;
  storeListingVersionId: string;
}

export const RatingCard: React.FC<RatingCardProps> = ({
  agentName,
  storeListingVersionId,
}) => {
  const [rating, setRating] = React.useState<number>(0);
  const [hoveredRating, setHoveredRating] = React.useState<number>(0);
  const [isVisible, setIsVisible] = React.useState(true);
  const api = useBackendAPI();

  const handleClose = () => {
    setIsVisible(false);
  };

  if (!isVisible) return null;

  const handleSubmit = async (rating: number) => {
    if (rating > 0) {
      console.log(`Rating submitted for ${agentName}:`, rating);
      await api.reviewAgent("--", agentName, {
        store_listing_version_id: storeListingVersionId,
        score: rating,
      });
      handleClose();
    }
  };

  const getRatingText = (rating: number) => {
    switch (rating) {
      case 1:
        return "Needs improvement";
      case 2:
        return "Meh";
      case 3:
        return "Average";
      case 4:
        return "Good";
      case 5:
        return "Awesome!";
      default:
        return "Rate it!";
    }
  };

  return (
    <div className="relative flex w-[400px] flex-col rounded-[32px] bg-white p-8 shadow-lg">
      <button
        onClick={handleClose}
        className="absolute right-6 top-6 h-6 w-6 text-gray-400 transition-colors hover:text-gray-600"
      >
        <Cross1Icon className="h-5 w-5" />
      </button>

      <div className="mb-2 text-center text-xl font-semibold">Rate agent</div>

      <div className="mb-6 text-center text-sm text-gray-600">
        Could you rate {agentName} agent for us?
      </div>

      <div className="mb-6 flex flex-col items-center">
        <div className="mb-2 flex gap-2">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              className="flex h-8 w-8 items-center justify-center"
              onMouseEnter={() => setHoveredRating(star)}
              onMouseLeave={() => setHoveredRating(0)}
              onClick={() => setRating(star)}
            >
              {star <= (hoveredRating || rating) ? (
                <IconStarFilled className="h-8 w-8 text-yellow-400" />
              ) : (
                <IconStar className="h-8 w-8 text-gray-200" />
              )}
            </button>
          ))}
        </div>

        <div className="text-center text-sm text-gray-600">
          {getRatingText(hoveredRating || rating)}
        </div>
      </div>

      <div className="self-end">
        <button
          onClick={() => rating > 0 && handleSubmit(rating)}
          className={`rounded-full px-6 py-2 text-sm font-medium transition-colors ${
            rating > 0
              ? "cursor-pointer bg-slate-900 text-white hover:bg-slate-800"
              : "cursor-not-allowed bg-gray-200 text-gray-400"
          }`}
        >
          Submit
        </button>
      </div>
    </div>
  );
};
