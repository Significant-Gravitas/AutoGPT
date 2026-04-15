import { useCallback, useEffect, useRef, useState } from "react";

interface useFilterChipsProps {
  onFilterChange?: (selectedFilters: string[]) => void;
  multiSelect?: boolean;
}

export function useFilterChips({
  onFilterChange,
  multiSelect,
}: useFilterChipsProps) {
  const [selectedFilters, setSelectedFilters] = useState<string[]>([]);
  const pendingFilters = useRef<string[] | null>(null);

  useEffect(() => {
    if (pendingFilters.current !== null) {
      onFilterChange?.(pendingFilters.current);
      pendingFilters.current = null;
    }
  }, [selectedFilters, onFilterChange]);

  const handleBadgeClick = useCallback(
    (badge: string) => {
      setSelectedFilters((prev) => {
        let next;
        if (multiSelect) {
          next = prev.includes(badge)
            ? prev.filter((f) => f !== badge)
            : [...prev, badge];
        } else {
          next = prev.includes(badge) ? [] : [badge];
        }
        pendingFilters.current = next;
        return next;
      });
    },
    [multiSelect],
  );

  return { selectedFilters, handleBadgeClick };
}
