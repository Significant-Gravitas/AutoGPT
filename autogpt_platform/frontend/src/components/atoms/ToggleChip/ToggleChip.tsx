"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ArrowDataTransferVerticalIcon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";

interface Props {
  icon: ReactNode;
  label: string;
  tooltip: string;
  ariaLabel: string;
  pressed: boolean;
  onToggle: () => void;
  locked?: boolean;
  className?: string;
}

export function ToggleChip({
  icon,
  label,
  tooltip,
  ariaLabel,
  pressed,
  onToggle,
  locked = false,
  className,
}: Props) {
  const prefersReducedMotion = useReducedMotion();
  // Blur bridges the two label states so the swap reads as one motion rather
  // than a hard cut; opacity alone carries it under reduced motion.
  const blurred = prefersReducedMotion ? "blur(0px)" : "blur(3px)";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-pressed={pressed}
          aria-disabled={locked}
          onClick={onToggle}
          aria-label={ariaLabel}
          className={cn(
            "group inline-flex h-7 items-center justify-center gap-1 rounded-full px-2.5 text-xs font-medium text-zinc-950 transition-colors hover:bg-white",
            locked && "cursor-not-allowed opacity-70 hover:bg-transparent",
            className,
          )}
        >
          <span className="relative inline-flex size-3.5 items-center justify-center">
            <span
              className={cn(
                "transition-opacity duration-150 ease-out",
                !locked && "group-hover:opacity-0",
              )}
            >
              {icon}
            </span>
            {!locked && (
              <Icon
                icon={ArrowDataTransferVerticalIcon}
                size={14}
                className="absolute opacity-0 transition-opacity duration-150 ease-out group-hover:opacity-100"
              />
            )}
          </span>
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={label}
              initial={{ filter: blurred, opacity: 0 }}
              animate={{ filter: "blur(0px)", opacity: 1 }}
              exit={{ filter: blurred, opacity: 0 }}
              transition={{ duration: 0.12, ease: [0, 0, 0.2, 1] }}
              className="hidden sm:inline"
            >
              {label}
            </motion.span>
          </AnimatePresence>
        </button>
      </TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  );
}
