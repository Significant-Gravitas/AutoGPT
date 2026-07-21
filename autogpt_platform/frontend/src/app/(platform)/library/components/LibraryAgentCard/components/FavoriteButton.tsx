"use client";

import { cn } from "@/lib/utils";
import { HeartIcon } from "@phosphor-icons/react";
import type { MouseEvent } from "react";
import { useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface FavoriteButtonProps {
  isFavorite: boolean;
  onClick: (
    e: MouseEvent<HTMLButtonElement>,
    position: { x: number; y: number },
  ) => void;
  className?: string;
}

export function FavoriteButton({
  isFavorite,
  onClick,
  className,
}: FavoriteButtonProps) {
  const buttonRef = useRef<HTMLButtonElement>(null);

  function handleClick(e: MouseEvent<HTMLButtonElement>) {
    const rect = buttonRef.current?.getBoundingClientRect();
    const position = rect
      ? {
          x: rect.left + rect.width / 2 - 12,
          y: rect.top + rect.height / 2 - 12,
        }
      : { x: 0, y: 0 };
    onClick(e, position);
  }

  return (
    <button
      ref={buttonRef}
      onClick={handleClick}
      className={cn(
        "rounded-full p-2 transition-all duration-200",
        "hover:scale-110 active:scale-95",
        !isFavorite && "opacity-0 group-hover:opacity-100",
        className,
      )}
      aria-label={isFavorite ? "Remove from favorites" : "Add to favorites"}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={isFavorite ? "filled" : "empty"}
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.5, opacity: 0 }}
          transition={{ type: "spring", damping: 15, stiffness: 300 }}
        >
          <HeartIcon
            size={20}
            weight={isFavorite ? "fill" : "regular"}
            className={cn(
              "transition-colors duration-200",
              isFavorite ? "text-red-500" : "text-gray-600 hover:text-red-500",
            )}
          />
        </motion.div>
      </AnimatePresence>
    </button>
  );
}
