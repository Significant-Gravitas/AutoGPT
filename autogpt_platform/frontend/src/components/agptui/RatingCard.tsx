"use client";

import * as React from "react";
import { Cross1Icon } from "@radix-ui/react-icons";
import { IconStar, IconStarFilled } from "@/components/ui/icons";
import { createClient } from "@/lib/supabase/client";
import AutoGPTServerAPI from "@/lib/autogpt-server-api/client";

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

  const supabase = createClient();

  const api = new AutoGPTServerAPI(
        process.env.NEXT_PUBLIC_AGPT_SERVER_URL,
        process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL,
        supabase,
  );

  const handleClose = () => {
    setIsVisible(false);
  };

  if (!isVisible) return null;

  const handleSubmit = async (rating: number) => {
    if (rating > 0) {
        console.log(`Rating submitted for ${agentName}:`, rating);
        await api.reviewAgent("--", agentName, {
          store_listing_version_id: storeListingVersionId,
          score: rating
        });
        handleClose();
    }
  };

  const getRatingText = (rating: number) => {
    switch (rating) {
      case 1: return "Needs improvement";
      case 2: return "Meh";
      case 3: return "Average";
      case 4: return "Good";
      case 5: return "Awesome!";
      default: return "Rate it!";
    }
  };

  return (
    <div className="w-[400px] p-8 bg-white rounded-[32px] shadow-lg flex flex-col relative">
      <button 
        onClick={handleClose}
        className="absolute top-6 right-6 w-6 h-6 text-gray-400 hover:text-gray-600 transition-colors"
      >
        <Cross1Icon className="w-5 h-5" />
      </button>
      
      <div className="text-xl font-semibold text-center mb-2">
        Rate agent
      </div>

      <div className="text-gray-600 text-sm text-center mb-6">
        Could you rate {agentName} agent for us?
      </div>
      
      <div className="flex flex-col items-center mb-6">
        <div className="flex gap-2 mb-2">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              className="w-8 h-8 flex items-center justify-center"
              onMouseEnter={() => setHoveredRating(star)}
              onMouseLeave={() => setHoveredRating(0)}
              onClick={() => setRating(star)}
            >
              {star <= (hoveredRating || rating) ? (
                <IconStarFilled className="w-8 h-8 text-yellow-400" />
              ) : (
                <IconStar className="w-8 h-8 text-gray-200" />
              )}
            </button>
          ))}
        </div>

        <div className="text-gray-600 text-sm text-center">
          {getRatingText(hoveredRating || rating)}
        </div>
      </div>

      <div className="self-end">
        <button
          onClick={() => rating > 0 && handleSubmit(rating)}
          className={`px-6 py-2 rounded-full text-sm font-medium transition-colors
            ${rating > 0 
              ? "bg-slate-900 text-white hover:bg-slate-800 cursor-pointer" 
              : "bg-gray-200 text-gray-400 cursor-not-allowed"
            }`}
        >
          Submit
        </button>
      </div>
    </div>
  );
};