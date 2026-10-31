import React, { Fragment } from "react";
import { Block } from "../Block";
import { beautifyString, cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";
import { Separator } from "@/components/atoms/Separator/Separator";

import { useAllBlockContent } from "./useAllBlockContent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { blockMenuButtonBase, blockMenuContainerStyle } from "../style";

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
          {index > 0 && (
            <Separator className="h-[1px] w-full text-foreground-disabled" />
          )}

          {/* Category Section */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <Text variant="body-medium" tone="emphasis">
                {category.name && beautifyString(category.name)}
              </Text>
              <Text
                variant="body"
                as="span"
                tone="secondary"
                className="rounded-full bg-surface-hover px-1.5"
              >
                {category.total_blocks}
              </Text>
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
                <button
                  type="button"
                  className={cn(
                    blockMenuButtonBase,
                    "h-9 py-2 text-foreground-secondary underline-offset-4 hover:text-foreground-emphasis hover:underline",
                  )}
                  disabled={isLoadingMore(category.name)}
                  onClick={() => handleRefetchBlocks(category.name)}
                >
                  <Text
                    variant="body-medium"
                    as="span"
                    tone="inherit"
                    className="underline"
                  >
                    see all
                  </Text>
                </button>
              )}
            </div>
          </div>
        </Fragment>
      ))}
    </div>
  );
};
