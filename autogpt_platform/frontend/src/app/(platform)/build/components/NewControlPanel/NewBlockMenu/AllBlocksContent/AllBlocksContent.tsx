import React, { Fragment } from "react";
import { Block } from "../Block";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Separator } from "@/components/atoms/Separator/Separator";
import { beautifyString } from "@/lib/utils";
import { useAllBlockContent } from "./useAllBlockContent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { blockMenuContainerStyle } from "../style";

export const AllBlocksContent = () => {
  const {
    data,
    isLoading,
    isError,
    error,
    handleRefetchBlocks,
    isLoadingMore,
    isErrorOnLoadingMore,
  } = useAllBlockContent();

  if (isLoading) {
    return (
      <div className={blockMenuContainerStyle}>
        {[0, 1, 2, 3, 4].map((skeletonIndex) => (
          <Block.Skeleton key={`skeleton-${skeletonIndex}`} />
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <div className="h-full p-4">
        <ErrorCard
          isSuccess={false}
          responseError={error || undefined}
          httpError={{
            status: data?.status,
            statusText: "Request failed",
            message: (error?.detail as string) || "An error occurred",
          }}
          context="block menu"
          onRetry={() => window.location.reload()}
        />
      </div>
    );
  }

  const categories = data?.categories;

  return (
    <div className={blockMenuContainerStyle}>
      {categories?.map((category, index) => (
        <Fragment key={category.name}>
          {index > 0 && <Separator className="h-px w-full text-zinc-300" />}

          {/* Category Section */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <Text variant="body-medium" className="text-zinc-800">
                {category.name && beautifyString(category.name)}
              </Text>
              <span className="rounded-full bg-zinc-100 px-1.5 font-sans text-sm leading-5.5 text-zinc-600">
                {category.total_blocks}
              </span>
            </div>

            <div className="space-y-2">
              {category.blocks.map((block) => (
                <Block
                  key={`${category.name}-${block.id}`}
                  title={block.name as string}
                  description={block.name as string}
                  blockData={block}
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

              {!isErrorOnLoadingMore && (
                <ErrorCard
                  isSuccess={false}
                  responseError={{ message: "Error loading blocks" }}
                  context="blocks"
                  onRetry={() => handleRefetchBlocks(category.name)}
                />
              )}

              {category.total_blocks > category.blocks.length && (
                <Button
                  variant="link"
                  className="h-9 min-w-0 px-0 py-2 font-sans text-sm leading-5.5 font-normal text-zinc-600 underline underline-offset-4 hover:text-zinc-800"
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
  );
};
