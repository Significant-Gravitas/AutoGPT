// Used v0 for this component, so review it very carefully

import FilterChip from "../FilterChip";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Separator } from "@/components/ui/separator";
import { CategoryKey, Filters } from "./FiltersList";
import { Checkbox } from "@/components/ui/checkbox";

export default function FilterSheet({
  filters,
  creators,
  onCategoryChange,
  onCreatorChange,
  categories,
}: {
  filters: Filters;
  creators: string[];
  onCategoryChange: (category: CategoryKey) => void;
  onCreatorChange: (creator: string) => void;
  categories: Array<{ key: CategoryKey; name: string }>;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [isSheetVisible, setIsSheetVisible] = useState(false);

  // Animation
  useEffect(() => {
    if (isOpen) {
      setIsSheetVisible(true);
    } else {
      const timer = setTimeout(() => {
        setIsSheetVisible(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  return (
    <div className="m-0 inline w-fit p-0">
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

            <div className="px-5">
              <p className="font-sans text-base font-medium text-zinc-800">
                Categories
              </p>
              <div className="mt-2 space-y-2">
                {categories.map((category) => (
                  <div
                    key={category.key}
                    className="flex items-center space-x-2"
                  >
                    <Checkbox
                      id={category.key}
                      checked={filters.categories[category.key]}
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

            <div className="px-5">
              <p className="font-sans text-base font-medium text-zinc-800">
                Created by
              </p>
              <div className="mt-2 space-y-2">
                {creators.map((creator) => (
                  <div key={creator} className="flex items-center space-x-2">
                    <Checkbox
                      id={`creator-${creator}`}
                      checked={filters.createdBy.includes(creator)}
                      onCheckedChange={() => onCreatorChange(creator)}
                      className="border border-[#D4D4D4] shadow-none data-[state=checked]:bg-white data-[state=checked]:text-zinc-500"
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
            </div>
          </div>
        </>
      )}
    </div>
  );
}
