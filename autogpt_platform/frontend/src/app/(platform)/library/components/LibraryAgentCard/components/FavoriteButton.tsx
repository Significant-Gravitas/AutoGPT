"use client";

import { cn } from "@/lib/utils";
import { HeartIcon } from "@phosphor-icons/react";
import type { MouseEvent } from "react";

interface FavoriteButtonProps {
  isFavorite: boolean;
  onClick: (e: MouseEvent<HTMLButtonElement>) => void;
  className?: string;
}

export function FavoriteButton({
  isFavorite,
  onClick,
  className,
}: FavoriteButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "rounded-full p-2 transition-all duration-200",
        "hover:scale-110",
        !isFavorite && "opacity-0 group-hover:opacity-100",
        className,
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
