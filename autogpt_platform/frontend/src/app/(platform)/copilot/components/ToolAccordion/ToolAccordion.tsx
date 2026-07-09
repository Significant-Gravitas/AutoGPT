"use client";

import { cn } from "@/lib/utils";
import { CaretDownIcon } from "@/components/atoms/AGPTIcon/icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useId } from "react";
import { useToolAccordion } from "./useToolAccordion";

interface Props {
  icon: React.ReactNode;
  title: React.ReactNode;
  titleClassName?: string;
  description?: React.ReactNode;
  descriptionClassName?: string;
  children: React.ReactNode;
  className?: string;
  defaultExpanded?: boolean;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  variant?: "card" | "compact";
}

export function ToolAccordion({
  icon,
  title,
  titleClassName,
  description,
  descriptionClassName,
  children,
  className,
  defaultExpanded,
  expanded,
  onExpandedChange,
  variant = "card",
}: Props) {
  const shouldReduceMotion = useReducedMotion();
  const contentId = useId();
  const { isExpanded, toggle } = useToolAccordion({
    expanded,
    defaultExpanded,
    onExpandedChange,
  });

  const isCompact = variant === "compact";

  return (
    <div
      className={cn(
        "w-full transition-colors",
        isCompact
          ? cn("mt-1 rounded-xl py-1 pr-2", isExpanded && "pb-2")
          : "mt-2 rounded-lg bg-stone-50 px-3 py-2",
        className,
      )}
    >
      <button
        type="button"
        aria-expanded={isExpanded}
        aria-controls={contentId}
        onClick={toggle}
        className={cn(
          "flex w-full items-center justify-between text-left",
          isCompact
            ? "gap-2 rounded-lg py-0.5 hover:bg-stone-50"
            : "gap-3 py-1",
        )}
      >
        <div
          className={cn(
            "flex min-w-0 items-center",
            isCompact ? "gap-2" : "gap-3",
          )}
        >
          <span className="flex shrink-0 items-center text-gray-800">
            {icon}
          </span>
          <div
            className={cn("min-w-0", isCompact && "flex items-baseline gap-2")}
          >
            <p
              className={cn(
                "truncate",
                isCompact
                  ? "text-xs text-gray-800"
                  : "text-sm font-medium text-gray-800",
                titleClassName,
              )}
            >
              {title}
            </p>
            {description && (
              <p
                className={cn(
                  "truncate text-xs",
                  isCompact ? "text-slate-500" : "text-slate-800",
                  descriptionClassName,
                )}
              >
                {description}
              </p>
            )}
          </div>
        </div>
        <CaretDownIcon
          className={cn(
            "shrink-0 text-slate-500 transition-transform",
            isCompact ? "h-3 w-3" : "h-4 w-4",
            isExpanded && "rotate-180",
          )}
          weight="bold"
        />
      </button>

      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            id={contentId}
            initial={{ height: 0, opacity: 0, filter: "blur(4px)" }}
            animate={{ height: "auto", opacity: 1, filter: "blur(0px)" }}
            exit={{ height: 0, opacity: 0, filter: "blur(4px)" }}
            transition={
              shouldReduceMotion
                ? { duration: 0 }
                : { duration: 0.4, ease: [0.16, 1, 0.3, 1] }
            }
            className="overflow-hidden"
            style={{ willChange: "height, opacity, filter" }}
          >
            <div
              className={cn(
                "max-h-[24rem] overflow-y-auto",
                isCompact ? "px-1 pb-1 pt-2" : "pb-2 pt-3",
              )}
            >
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
