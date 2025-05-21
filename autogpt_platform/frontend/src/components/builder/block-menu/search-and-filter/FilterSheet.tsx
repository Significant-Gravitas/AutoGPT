import FilterChip from "../FilterChip";
import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import {
  CategoryKey,
  Filters,
  useBlockMenuContext,
} from "../block-menu-provider";

export default function FilterSheet({
  categories,
}: {
  categories: Array<{ key: CategoryKey; name: string }>;
}) {
  const { filters, creators, setFilters } = useBlockMenuContext();
  const [isOpen, setIsOpen] = useState(false);
  const [isSheetVisible, setIsSheetVisible] = useState(false);
  const [localFilters, setLocalFilters] = useState<Filters>(filters);

  // Animation
  useEffect(() => {
    if (isOpen) {
      setIsSheetVisible(true);
      setLocalFilters(filters);
    } else {
      const timer = setTimeout(() => {
        setIsSheetVisible(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen, filters]);

  const onCategoryChange = useCallback((category: CategoryKey) => {
    setLocalFilters((prev) => ({
      ...prev,
      categories: {
        ...prev.categories,
        [category]: !prev.categories[category],
      },
    }));
  }, []);

  const onCreatorChange = useCallback((creator: string) => {
    setLocalFilters((prev) => {
      const updatedCreators = prev.createdBy.includes(creator)
        ? prev.createdBy.filter((c) => c !== creator)
        : [...prev.createdBy, creator];

      return {
        ...prev,
        createdBy: updatedCreators,
      };
    });
  }, []);

  const handleApplyFilters = useCallback(() => {
    setFilters(localFilters);
    setIsOpen(false);
  }, [localFilters, setFilters]);

  const handleClearFilters = useCallback(() => {
    const clearedFilters: Filters = {
      categories: {
        blocks: false,
        integrations: false,
        marketplace_agents: false,
        my_agents: false,
        templates: false,
      },
      createdBy: [],
    };
    setFilters(clearedFilters);
    setIsOpen(false);
  }, [setFilters]);

  const hasActiveFilters = useCallback(() => {
    const hasCategoryFilter = Object.values(localFilters.categories).some(
      (value) => value,
    );
    const hasCreatorFilter = localFilters.createdBy.length > 0;

    return hasCategoryFilter || hasCreatorFilter;
  }, [localFilters]);

  return (
    <div className="m-0 ml-4 inline w-fit p-0">
      <Button
        onClick={() => {
          setIsSheetVisible(true);
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              setIsOpen(true);
            });
          });
        }}
        variant={"link"}
        className="m-0 p-0 hover:no-underline"
      >
        <FilterChip name="All filters" />
      </Button>

      {isSheetVisible && (
        <>
          <div
            className={cn(
              "absolute bottom-2 left-2 top-2 z-20 w-3/4 max-w-[22.5rem] space-y-4 rounded-[0.75rem] bg-white py-4 shadow-[0_4px_12px_2px_rgba(0,0,0,0.1)] transition-all",
              isOpen
                ? "translate-x-0 duration-300 ease-out"
                : "-translate-x-full duration-300 ease-out",
            )}
          >
            {/* Top */}
            <div className="flex items-center justify-between px-5">
              <p className="font-sans text-base text-[#040404]">Filters</p>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(false)}
              >
                <X className="h-5 w-5" />
              </Button>
            </div>

            <Separator className="h-[1px] w-full text-zinc-300" />

            {/* Categories */}

            <div className="space-y-4 px-5">
              <p className="font-sans text-base font-medium text-zinc-800">
                Categories
              </p>
              <div className="space-y-2">
                {categories.map((category) => (
                  <div
                    key={category.key}
                    className="flex items-center space-x-2"
                  >
                    <Checkbox
                      id={category.key}
                      checked={localFilters.categories[category.key]}
                      onCheckedChange={() => onCategoryChange(category.key)}
                      className="border border-[#D4D4D4] shadow-none data-[state=checked]:border-none data-[state=checked]:bg-zinc-500 data-[state=checked]:text-white"
                    />
                    <label
                      htmlFor={category.key}
                      className="font-sans text-sm leading-[1.375rem] text-zinc-600"
                    >
                      {category.name}
                    </label>
                  </div>
                ))}
              </div>
            </div>

            <Separator className="h-[1px] w-full text-zinc-300" />

            {/* Created By */}

            <div className="space-y-4 px-5">
              <p className="font-sans text-base font-medium text-zinc-800">
                Created by
              </p>
              <div className="space-y-2">
                {creators.map((creator) => (
                  <div key={creator} className="flex items-center space-x-2">
                    <Checkbox
                      id={`creator-${creator}`}
                      checked={localFilters.createdBy.includes(creator)}
                      onCheckedChange={() => onCreatorChange(creator)}
                      className="border border-[#D4D4D4] shadow-none data-[state=checked]:border-none data-[state=checked]:bg-zinc-500 data-[state=checked]:text-white"
                    />
                    <label
                      htmlFor={`creator-${creator}`}
                      className="font-sans text-sm leading-[1.375rem] text-zinc-600"
                    >
                      {creator}
                    </label>
                  </div>
                ))}
              </div>
              <Button
                variant={"link"}
                className="m-0 p-0 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800 underline hover:text-zinc-600"
              >
                More
              </Button>
            </div>

            {/* Footer buttons */}
            <div className="absolute bottom-0 flex w-full justify-between gap-3 border-t border-zinc-300 px-5 py-3">
              <Button
                className="min-w-[5rem] rounded-[0.5rem] border-none px-1.5 py-2 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800 shadow-none ring-1 ring-zinc-400"
                variant={"outline"}
                onClick={handleClearFilters}
              >
                Clear
              </Button>

              <Button
                className={cn(
                  "min-w-[6.25rem] rounded-[0.5rem] border-none px-1.5 py-2 font-sans text-sm font-medium leading-[1.375rem] text-white shadow-none ring-1 disabled:ring-0",
                )}
                onClick={handleApplyFilters}
                disabled={!hasActiveFilters()}
              >
                Apply filters
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
