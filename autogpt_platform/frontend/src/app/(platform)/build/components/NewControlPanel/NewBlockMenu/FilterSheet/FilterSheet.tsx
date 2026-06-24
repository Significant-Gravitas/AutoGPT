import { FilterChip } from "../FilterChip";
import { cn } from "@/lib/utils";
import { CategoryKey } from "../BlockMenuFilters/types";
import { AnimatePresence, motion } from "framer-motion";
import { XIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Separator } from "@/components/__legacy__/ui/separator";
import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import { useFilterSheet } from "./useFilterSheet";
import { INITIAL_CREATORS_TO_SHOW } from "./constant";

export function FilterSheet({
  categories,
}: {
  categories: Array<{ key: CategoryKey; name: string }>;
}) {
  const {
    isOpen,
    localCategories,
    localCreators,
    displayedCreatorsCount,
    handleLocalCategoryChange,
    handleToggleShowMoreCreators,
    handleLocalCreatorChange,
    handleClearFilters,
    handleCloseButton,
    handleApplyFilters,
    hasLocalActiveFilters,
    visibleCreators,
    creators,
    handleOpenFilters,
    hasActiveFilters,
  } = useFilterSheet();

  return (
    <div className="m-0 inline w-fit p-0">
      <FilterChip
        name={hasActiveFilters() ? "Edit filters" : "All filters"}
        onClick={handleOpenFilters}
      />

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className={cn(
              "absolute bottom-2 left-2 top-2 z-20 w-3/4 max-w-[22.5rem] space-y-4 overflow-hidden rounded-[0.75rem] bg-white pb-4 shadow-[0_4px_12px_2px_rgba(0,0,0,0.1)]",
            )}
            initial={{ x: "-100%", filter: "blur(10px)" }}
            animate={{ x: 0, filter: "blur(0px)" }}
            exit={{ x: "-110%", filter: "blur(10px)" }}
            transition={{ duration: 0.4, type: "spring", bounce: 0.2 }}
          >
            {/* Top section */}
            <div className="flex items-center justify-between px-5 pt-4">
              <Text variant="body">Filters</Text>
              <Button
                className="p-0"
                variant="ghost"
                size="icon"
                onClick={handleCloseButton}
              >
                <XIcon size={20} />
              </Button>
            </div>

            <Separator className="h-[1px] w-full text-zinc-300" />

            {/* Category section */}
            <div className="space-y-4 px-5">
              <Text variant="large">Categories</Text>
              <div className="space-y-2">
                {categories.map((category) => (
                  <div
                    key={category.key}
                    className="flex items-center space-x-2"
                  >
                    <Checkbox
                      id={category.key}
                      checked={localCategories.includes(category.key)}
                      onCheckedChange={() =>
                        handleLocalCategoryChange(category.key)
                      }
                      className="border border-[#D4D4D4] shadow-none data-[state=checked]:border-none data-[state=checked]:bg-violet-700 data-[state=checked]:text-white"
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

            {/* Created by section */}
            <div className="space-y-4 px-5">
              <p className="font-sans text-base font-medium text-zinc-800">
                Created by
              </p>
              <div className="space-y-2">
                {visibleCreators.map((creator, i) => (
                  <div key={i} className="flex items-center space-x-2">
                    <Checkbox
                      id={`creator-${creator}`}
                      checked={localCreators.includes(creator)}
                      onCheckedChange={() => handleLocalCreatorChange(creator)}
                      className="border border-[#D4D4D4] shadow-none data-[state=checked]:border-none data-[state=checked]:bg-violet-700 data-[state=checked]:text-white"
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
              {creators.length > INITIAL_CREATORS_TO_SHOW && (
                <Button
                  variant={"link"}
                  className="m-0 p-0 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800 underline hover:text-zinc-600"
                  onClick={handleToggleShowMoreCreators}
                >
                  {displayedCreatorsCount < creators.length ? "More" : "Less"}
                </Button>
              )}
            </div>

            {/* Footer section */}
            <div className="fixed bottom-0 flex w-full justify-between gap-3 border-t border-zinc-200 bg-white px-5 py-3">
              <Button
                size="small"
                variant={"outline"}
                onClick={handleClearFilters}
                className="rounded-[8px] px-2 py-1.5"
              >
                Clear
              </Button>

              <Button
                size="small"
                onClick={handleApplyFilters}
                disabled={!hasLocalActiveFilters()}
                className="rounded-[8px] px-2 py-1.5"
              >
                Apply filters
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
