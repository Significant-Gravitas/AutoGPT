import React, { useState, useEffect, Fragment } from "react";
import Block from "../Block";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { BlockCategoryResponse } from "@/lib/autogpt-server-api";
import { useBlockMenuContext } from "../block-menu-provider";

const AllBlocksContent: React.FC = () => {
  const { addNode } = useBlockMenuContext();
  const [categories, setCategories] = useState<BlockCategoryResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingCategories, setLoadingCategories] = useState<Set<string>>(
    new Set(),
  );

  const api = useBackendAPI();

  useEffect(() => {
    const fetchBlocks = async () => {
      setLoading(true);
      const response = await api.getBlockCategories();
      setCategories(response);
      setLoading(false);
    };

    fetchBlocks();
  }, [api]);

  const fetchMoreBlockOfACategory = async (category: string) => {
    try {
      setLoadingCategories((prev) => new Set(prev).add(category));
      const response = await api.getBuilderBlocks({ category: category });
      const updatedCategories = categories.map((cat) => {
        if (cat.name === category) {
          return {
            ...cat,
            blocks: [...response.blocks],
          };
        }
        return cat;
      });

      setCategories(updatedCategories);
    } catch (error) {
      console.error(`Failed to fetch blocks for category ${category}:`, error);
    } finally {
      setLoadingCategories((prev) => {
        const newSet = new Set(prev);
        newSet.delete(category);
        return newSet;
      });
    }
  };

  if (loading) {
    return (
      <div className="w-full space-y-3 p-4">
        {[0, 1, 3].map((categoryIndex) => (
          <Fragment key={categoryIndex}>
            {categoryIndex > 0 && (
              <Skeleton className="my-4 h-[1px] w-full text-zinc-100" />
            )}
            {[0, 1, 2].map((blockIndex) => (
              <Block.Skeleton key={`${categoryIndex}-${blockIndex}`} />
            ))}
          </Fragment>
        ))}
      </div>
    );
  }

  return (
    <div className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200 transition-all duration-200">
      <div className="w-full space-y-3 px-4 pb-4">
        {categories.map((category, index) => (
          <Fragment key={category.name}>
            {index > 0 && (
              <Separator className="h-[1px] w-full text-zinc-300" />
            )}

            {/* Category Section */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
                  {category.name}
                </p>
                <span className="rounded-full bg-zinc-100 px-[0.375rem] font-sans text-sm leading-[1.375rem] text-zinc-600">
                  {category.total_blocks}
                </span>
              </div>

              <div className="space-y-2">
                {category.blocks.map((block, idx) => (
                  // It might go fail for agent block
                  <Block
                    key={`${category.name}-${idx}`}
                    title={block.name}
                    description={block.name}
                    onClick={() => {
                      addNode(
                        block.id,
                        block.name,
                        block.hardcodedValues || {},
                        block,
                      );
                    }}
                  />
                ))}

                {loadingCategories.has(category.name) && (
                  <>
                    {[0, 1, 2, 3, 4].map((skeletonIndex) => (
                      <Block.Skeleton
                        key={`skeleton-${category.name}-${skeletonIndex}`}
                      />
                    ))}
                  </>
                )}

                {category.total_blocks > category.blocks.length && (
                  <Button
                    variant={"link"}
                    className="px-0 font-sans text-sm leading-[1.375rem] text-zinc-600 underline hover:text-zinc-800"
                    disabled={loadingCategories.has(category.name)}
                    onClick={() => {
                      fetchMoreBlockOfACategory(category.name);
                    }}
                  >
                    see all
                  </Button>
                )}
              </div>
            </div>
          </Fragment>
        ))}
      </div>
    </div>
  );
};

export default AllBlocksContent;
