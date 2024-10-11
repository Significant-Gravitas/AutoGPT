import * as React from "react";
import { Badge } from "@/components/ui/badge";

interface FilterChipsProps {
  badges: string[];
  onFilterChange?: (selectedFilters: string[]) => void;
  multiSelect?: boolean;
}
/** FilterChips is a component that allows the user to select filters from a list of badges. It is used on the Agent Store home page */
export const FilterChips: React.FC<FilterChipsProps> = ({
  badges,
  onFilterChange,
  multiSelect = true,
}) => {
  const [selectedFilters, setSelectedFilters] = React.useState<string[]>([]);

  const handleBadgeClick = (badge: string) => {
    setSelectedFilters((prevFilters) => {
      let newFilters;
      if (multiSelect) {
        newFilters = prevFilters.includes(badge)
          ? prevFilters.filter((filter) => filter !== badge)
          : [...prevFilters, badge];
      } else {
        newFilters = prevFilters.includes(badge) ? [] : [badge];
      }

      if (onFilterChange) {
        onFilterChange(newFilters);
      }

      return newFilters;
    });
  };

  return (
    <div className="inline-flex h-14 items-center justify-start gap-5">
      {badges.map((badge) => (
        <Badge
          key={badge}
          variant={selectedFilters.includes(badge) ? "secondary" : "outline"}
          className="h-1] flex cursor-pointer items-center justify-center gap-2.5 rounded-full border border-black/50 px-6 py-2"
          onClick={() => handleBadgeClick(badge)}
        >
          <div className="font-neue text-xl font-medium leading-9 tracking-tight text-[#474747]">
            {badge}
          </div>
        </Badge>
      ))}
    </div>
  );
};
