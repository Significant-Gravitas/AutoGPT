"use client";

import { cn } from "@/lib/utils";
import { CaretDownIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useId } from "react";
import { useToolAccordion } from "./useToolAccordion";

interface Props {
  icon: React.ReactNode;
  title: React.ReactNode;
  titleClassName?: string;
  description?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  defaultExpanded?: boolean;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
}

export function ToolAccordion({
  icon,
  title,
  titleClassName,
  description,
  children,
  className,
  defaultExpanded,
  expanded,
  onExpandedChange,
}: Props) {
  const shouldReduceMotion = useReducedMotion();
  const contentId = useId();
  const { isExpanded, toggle } = useToolAccordion({
    expanded,
    defaultExpanded,
    onExpandedChange,
  });

  return (
    <div
      className={cn(
        "mt-2 w-full rounded-lg border border-slate-200 bg-slate-100 px-3 py-2",
        className,
      )}
    >
      <button
        type="button"
        aria-expanded={isExpanded}
        aria-controls={contentId}
        onClick={toggle}
        className="flex w-full items-center justify-between gap-3 py-1 text-left"
      >
        <div className="flex min-w-0 items-center gap-3">
          <span className="flex shrink-0 items-center text-gray-800">
            {icon}
          </span>
          <div className="min-w-0">
            <p
              className={cn(
                "truncate text-sm font-medium text-gray-800",
                titleClassName,
              )}
            >
              {title}
            </p>
            {description && (
              <p className="truncate text-xs text-slate-800">{description}</p>
            )}
          </div>
        </div>
        <CaretDownIcon
          className={cn(
            "h-4 w-4 shrink-0 text-slate-500 transition-transform",
            isExpanded && "rotate-180",
          )}
          weight="bold"
        />
      </button>

      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            id={contentId}
            initial={{ height: 0, opacity: 0, filter: "blur(10px)" }}
            animate={{ height: "auto", opacity: 1, filter: "blur(0px)" }}
            exit={{ height: 0, opacity: 0, filter: "blur(10px)" }}
            transition={
              shouldReduceMotion
                ? { duration: 0 }
                : { type: "spring", bounce: 0.35, duration: 0.55 }
            }
            className="overflow-hidden"
            style={{ willChange: "height, opacity, filter" }}
          >
            <div className="pb-2 pt-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
