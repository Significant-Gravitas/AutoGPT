import { useState } from "react";

interface UseToolAccordionOptions {
  expanded?: boolean;
  defaultExpanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
}

interface UseToolAccordionResult {
  isExpanded: boolean;
  toggle: () => void;
}

export function useToolAccordion({
  expanded,
  defaultExpanded = false,
  onExpandedChange,
}: UseToolAccordionOptions): UseToolAccordionResult {
  const [uncontrolledExpanded, setUncontrolledExpanded] =
    useState(defaultExpanded);

  const isControlled = typeof expanded === "boolean";
  const isExpanded = isControlled ? expanded : uncontrolledExpanded;

  function toggle() {
    const next = !isExpanded;
    if (!isControlled) setUncontrolledExpanded(next);
    onExpandedChange?.(next);
  }

  return { isExpanded, toggle };
}
