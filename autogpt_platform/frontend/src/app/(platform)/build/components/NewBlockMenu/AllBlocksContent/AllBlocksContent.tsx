import React, { Fragment } from "react";
import { Block } from "../Block";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { beautifyString } from "@/lib/utils";
import { scrollbarStyles } from "@/components/styles/scrollbar";
import { useAllBlockContent } from "./useAllBlockContent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

export const AllBlocksContent = () => {
  const {data, isLoading, isError, error, handleRefetchBlocks, isLoadingMore, isErrorOnLoadingMore} = useAllBlockContent();

  if (isLoading) {
    return (
      <div className={scrollbarStyles}>
        <div className="w-full space-y-3 px-4 pb-4">
            {[0, 1, 2, 3, 4].map((skeletonIndex) => (
                <Block.Skeleton
                    key={`skeleton-${skeletonIndex}`}
                />
            ))}
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="h-full p-4">
        <ErrorCard
          isSuccess={false}
          responseError={{ message: error?.detail as string }}
          context="blocks"
          onRetry={() => window.location.reload()}
        />
      </div>
    );
  }

  return (
    <div className={scrollbarStyles}>
      <div className="w-full space-y-3 px-4 pb-4">
        {data?.map((category, index) => (
          <Fragment key={category.name}>
            {index > 0 && (
              <Separator className="h-[1px] w-full text-zinc-300" />
            )}

            {/* Category Section */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
                  {category.name && beautifyString(category.name)}
                </p>
                <span className="rounded-full bg-zinc-100 px-[0.375rem] font-sans text-sm leading-[1.375rem] text-zinc-600">
                  {category.total_blocks}
                </span>
              </div>

              <div className="space-y-2">
                {category.blocks.map((block) => (
                  <Block
                    key={`${category.name}-${block.id}`}
                    title={block.name as string}
                    description={block.name as string}
                  />
                ))}

                {isLoadingMore(category.name) && (
                  <>
                    {[0, 1, 2].map((skeletonIndex) => (
                      <Block.Skeleton
                        key={`skeleton-${category.name}-${skeletonIndex}`}
                      />
                    ))}
                  </>
                )}

                {
                  !isErrorOnLoadingMore && (
                    <ErrorCard
                      isSuccess={false}
                      responseError={{ message: "Error loading blocks" }}
                      context="blocks"
                      onRetry={() => handleRefetchBlocks(category.name)}
                    />
                  )
                }

                {category.total_blocks > category.blocks.length && (
                  <Button
                    variant={"link"}
                    className="px-0 font-sans text-sm leading-[1.375rem] text-zinc-600 underline hover:text-zinc-800"
                    disabled={isLoadingMore(category.name)}
                    onClick={() => handleRefetchBlocks(category.name)}
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