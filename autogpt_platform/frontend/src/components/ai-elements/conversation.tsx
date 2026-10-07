"use client";

import { Button } from "@/components/atoms/Button/Button";
import type { ButtonProps } from "@/components/atoms/Button/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowDown02Icon } from "@hugeicons/core-free-icons";
import type { ComponentProps } from "react";
import { useCallback } from "react";
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
      "scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300",
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
        {icon && <div className="text-zinc-500">{icon}</div>}
        <div className="space-y-1">
          <Text variant="body-medium" as="h3">
            {title}
          </Text>
          {description && (
            <Text variant="body" tone="muted">
              {description}
            </Text>
          )}
        </div>
      </>
    )}
  </div>
);

export type ConversationScrollButtonProps = Extract<
  ButtonProps,
  { as?: "button" }
>;

export const ConversationScrollButton = ({
  className,
  ...props
}: ConversationScrollButtonProps) => {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  const handleScrollToBottom = useCallback(() => {
    scrollToBottom("instant");
  }, [scrollToBottom]);

  return (
    !isAtBottom && (
      <Button
        className={cn(
          "absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full",
          className,
        )}
        onClick={handleScrollToBottom}
        size="icon-sm"
        type="button"
        variant="secondary"
        aria-label="Scroll to bottom"
        withTooltip={false}
        {...props}
      >
        <Icon icon={ArrowDown02Icon} className="size-4" />
      </Button>
    )
  );
};
