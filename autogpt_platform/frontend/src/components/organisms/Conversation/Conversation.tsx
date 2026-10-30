"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { ArrowDown02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { ButtonHTMLAttributes, ComponentProps } from "react";

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
        {icon && <div className="text-foreground-tertiary">{icon}</div>}
        <div className="space-y-1">
          <Text variant="body-medium-tight" tone="inherit" as="h3">
            {title}
          </Text>
          {description && (
            <Text variant="body-tight" as="p" tone="muted">
              {description}
            </Text>
          )}
        </div>
      </>
    )}
  </div>
);

export type ConversationScrollButtonProps =
  ButtonHTMLAttributes<HTMLButtonElement>;

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
        size="compact-icon"
        type="button"
        variant="compact-outline"
        {...props}
      >
        <Icon aria-hidden icon={ArrowDown02Icon} className="size-4" />
      </Button>
    )
  );
};
