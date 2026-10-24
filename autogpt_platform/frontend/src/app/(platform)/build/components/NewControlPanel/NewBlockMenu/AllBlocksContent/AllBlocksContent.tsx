import React, { Fragment } from "react";
import { Block } from "../Block";
import { Button } from "@/components/__legacy__/ui/button";
import { Separator } from "@/components/__legacy__/ui/separator";
import { beautifyString } from "@/lib/utils";
import { useAllBlockContent } from "./useAllBlockContent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { blockMenuContainerStyle } from "../style";
import { Text } from "@/components/atoms/Text/Text";

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
                className="rounded-full bg-surface-hover px-[0.375rem]"
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
                <Button
                  variant={"link"}
                  className="px-0 font-sans text-body text-foreground-secondary underline hover:text-foreground-emphasis"
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
