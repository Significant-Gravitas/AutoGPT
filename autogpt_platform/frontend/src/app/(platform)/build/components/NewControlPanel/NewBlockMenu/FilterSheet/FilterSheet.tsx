import { FilterChip } from "../FilterChip";
import { cn } from "@/lib/utils";
import { CategoryKey } from "../BlockMenuFilters/types";
import { AnimatePresence, motion } from "framer-motion";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Separator } from "@/components/atoms/Separator/Separator";
import { Checkbox } from "@/components/atoms/Checkbox/Checkbox";
import { useFilterSheet } from "./useFilterSheet";
import { INITIAL_CREATORS_TO_SHOW } from "./constant";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
              // TODO(tokens): no exact role: shadow-[…] #000000
              "absolute bottom-2 left-2 top-2 z-20 w-3/4 max-w-[22.5rem] space-y-4 overflow-hidden rounded-[0.75rem] bg-surface pb-4 shadow-[0_4px_12px_2px_rgba(0,0,0,0.1)]",
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
                <Icon icon={Cancel01Icon} size={20} />
              </Button>
            </div>

            <Separator
              // TODO(tokens): no fitting role for text-zinc-300 #C5C5C9; #C5C5C9 is foreground-disabled (disabled text) and border-strong (a border colour), and this is a static divider (QUALITY 41(c))
              className="h-[1px] w-full text-zinc-300"
            />

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
                      // TODO(tokens): no exact role: border-[#D4D4D4] #D4D4D4; no fitting role for data-[state=checked]:bg-violet-700 #6D28D9; #6D28D9 is accent-strong (a text colour)
                      className="border border-[#D4D4D4] shadow-none data-[state=checked]:border-none data-[state=checked]:bg-violet-700 data-[state=checked]:text-action-foreground"
                    />
                    <Text
                      variant="body"
                      as="label"
                      tone="secondary"
                      htmlFor={category.key}
                    >
                      {category.name}
                    </Text>
                  </div>
                ))}
              </div>
            </div>

            {/* Created by section */}
            <div className="space-y-4 px-5">
              <Text variant="large-medium-tight" tone="emphasis">
                Created by
              </Text>
              <div className="space-y-2">
                {visibleCreators.map((creator, i) => (
                  <div key={i} className="flex items-center space-x-2">
                    <Checkbox
                      id={`creator-${creator}`}
                      checked={localCreators.includes(creator)}
                      onCheckedChange={() => handleLocalCreatorChange(creator)}
                      // TODO(tokens): no exact role: border-[#D4D4D4] #D4D4D4; no fitting role for data-[state=checked]:bg-violet-700 #6D28D9; #6D28D9 is accent-strong (a text colour)
                      className="border border-[#D4D4D4] shadow-none data-[state=checked]:border-none data-[state=checked]:bg-violet-700 data-[state=checked]:text-action-foreground"
                    />
                    <Text
                      variant="body"
                      as="label"
                      tone="secondary"
                      htmlFor={`creator-${creator}`}
                    >
                      {creator}
                    </Text>
                  </div>
                ))}
              </div>
              {creators.length > INITIAL_CREATORS_TO_SHOW && (
                <Button
                  variant={"link"}
                  className="m-0 p-0 text-foreground-emphasis hover:text-foreground-secondary"
                  onClick={handleToggleShowMoreCreators}
                >
                  <Text variant="body-medium" as="span" tone="inherit">
                    {displayedCreatorsCount < creators.length ? "More" : "Less"}
                  </Text>
                </Button>
              )}
            </div>

            {/* Footer section */}
            <div className="fixed bottom-0 flex w-full justify-between gap-3 border-t border-border-base bg-surface px-5 py-3">
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
