"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ArrowDownIcon } from "lucide-react";
import type { ComponentProps } from "react";

import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";

export type ConversationProps = ComponentProps<typeof StickToBottom>;

export const Conversation = ({ className, ...props }: ConversationProps) => (
  <StickToBottom
    className={cn("relative flex-1 overflow-y-hidden", className)}
    initial="instant"
    resize="instant"
    role="log"
    {...props}
  />
);

export type ConversationContentProps = ComponentProps<
  typeof StickToBottom.Content
>;

export const ConversationContent = ({
  className,
  scrollClassName,
  ...props
}: ConversationContentProps) => (
  <StickToBottom.Content
    className={cn("flex flex-col gap-8 p-4", className)}
    scrollClassName={cn(
      // Contain the overscroll so a fling past either end bounces the thread
      // instead of dragging the page, composer and all, with it.
      "overscroll-y-contain scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border-strong",
      scrollClassName,
    )}
    {...props}
  />
);

export type ConversationEmptyStateProps = ComponentProps<"div"> & {
  title?: string;
  description?: string;
  icon?: React.ReactNode;
};

export const ConversationEmptyState = ({
  className,
  title = "No messages yet",
  description = "Start a conversation to see messages here",
  icon,
  children,
  ...props
}: ConversationEmptyStateProps) => (
  <div
    className={cn(
      "flex size-full flex-col items-center justify-center gap-3 p-8 text-center",
      className,
    )}
    {...props}
  >
    {children ?? (
      <>
        {/* TODO(tokens): no exact role for text-neutral-500 #737373 */}
        {icon && <div className="text-neutral-500">{icon}</div>}
        <div className="space-y-1">
          {/* Text's 22px body line-height and #141414 would change these:
              the heading keeps 20px and the colour it inherits. */}
          <Text
            variant="body-medium"
            as="h3"
            className="leading-5 text-inherit"
          >
            {title}
          </Text>
          {/* TODO(tokens): no exact role for text-neutral-500 #737373 */}
          {description && (
            <Text variant="body" as="p" className="leading-5 text-neutral-500">
              {description}
            </Text>
          )}
        </div>
      </>
    )}
  </div>
);

export type ConversationScrollButtonProps = ComponentProps<typeof Button>;

export const ConversationScrollButton = ({
  className,
  ...props
}: ConversationScrollButtonProps) => {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  function handleScrollToBottom() {
    scrollToBottom("instant");
  }

  return (
    !isAtBottom && (
      <Button
        className={cn(
          "absolute bottom-4 left-[50%] translate-x-[-50%] rounded-full [@media(pointer:coarse)]:size-10",
          className,
        )}
        onClick={handleScrollToBottom}
        size="icon"
        type="button"
        variant="outline"
        {...props}
      >
        <ArrowDownIcon className="size-4" />
      </Button>
    )
  );
};
