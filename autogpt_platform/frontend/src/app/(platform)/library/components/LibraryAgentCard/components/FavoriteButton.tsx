"use client";

import { cn } from "@/lib/utils";
import { HeartIcon } from "@phosphor-icons/react";

interface FavoriteButtonProps {
  isFavorite: boolean;
  onClick: (e: React.MouseEvent) => void;
}

export function FavoriteButton({ isFavorite, onClick }: FavoriteButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "rounded-full bg-white/90 p-2 backdrop-blur-sm transition-all duration-200",
        "hover:scale-110 hover:bg-white",
        "focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2",
        !isFavorite && "opacity-0 group-hover:opacity-100",
      )}
      aria-label={isFavorite ? "Remove from favorites" : "Add to favorites"}
    >
      <HeartIcon
        size={20}
        weight={isFavorite ? "fill" : "regular"}
        className={cn(
          "transition-colors duration-200",
          isFavorite ? "text-red-500" : "text-gray-600 hover:text-red-500",
        )}
      />
    </button>
  );
}
