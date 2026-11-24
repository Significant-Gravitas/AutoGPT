"use client";

import { ButtonGroup } from "./components/ButtonGroup";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { lazyComponent } from "@/lib/lazy-component";
import { cn } from "@/lib/utils";
import type { UIMessage } from "ai";
import { ArrowLeft01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type {
  ButtonHTMLAttributes,
  ComponentProps,
  HTMLAttributes,
  ReactElement,
} from "react";
import { createContext, memo, useContext, useEffect, useState } from "react";
import type { Streamdown } from "streamdown";

export type MessageProps = HTMLAttributes<HTMLDivElement> & {
  from: UIMessage["role"];
};

const MessageRoleContext = createContext<MessageProps["from"]>("assistant");

export const Message = ({ className, from, ...props }: MessageProps) => (
  <MessageRoleContext value={from}>
    <div
      className={cn(
        "group flex w-full max-w-[95%] flex-col gap-2",
        // oxlint-disable-next-line design-system/no-unknown-classes -- is-user and is-assistant are markers the group-[.is-*] variants below match
        from === "user" ? "is-user ml-auto justify-end" : "is-assistant",
        className,
      )}
      {...props}
    />
  </MessageRoleContext>
);

export type MessageContentProps = HTMLAttributes<HTMLDivElement>;

export const MessageContent = ({
  children,
  className,
  ...props
}: MessageContentProps) => (
  <div
    className={cn(
      "flex w-full max-w-full min-w-0 flex-col gap-2 overflow-hidden text-body leading-5",
      "group-[.is-user]:w-fit",
      "group-[.is-user]:ml-auto group-[.is-user]:rounded-lg group-[.is-user]:bg-surface-hover group-[.is-user]:px-4 group-[.is-user]:py-3 group-[.is-user]:text-foreground-base",
      "group-[.is-assistant]:text-foreground-base",
      className,
    )}
    {...props}
  >
    {children}
  </div>
);

export type MessageActionsProps = ComponentProps<"div">;

export const MessageActions = ({
  className,
  children,
  ...props
}: MessageActionsProps) => (
  <div className={cn("flex items-center gap-1", className)} {...props}>
    {children}
  </div>
);

// Callers name the compact ghost icon look of the deleted ui/button by its old
// names; these maps keep that API.
const MESSAGE_ACTION_VARIANT = { ghost: "compact-ghost" } as const;
const MESSAGE_ACTION_SIZE = { "icon-sm": "compact-icon-sm" } as const;

export type MessageActionProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  tooltip?: string;
  label?: string;
  variant?: keyof typeof MESSAGE_ACTION_VARIANT;
  size?: keyof typeof MESSAGE_ACTION_SIZE;
};

export const MessageAction = ({
  tooltip,
  children,
  label,
  variant = "ghost",
  size = "icon-sm",
  className,
  ...props
}: MessageActionProps) => {
  const button = (
    <Button
      size={MESSAGE_ACTION_SIZE[size]}
      type="button"
      variant={MESSAGE_ACTION_VARIANT[variant]}
      // A finger needs a bigger target than the 32px icon box.
      className={cn("[@media(pointer:coarse)]:size-10", className)}
      {...props}
    >
      {children}
      <span className="sr-only">{label || tooltip}</span>
    </Button>
  );

  if (tooltip) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{button}</TooltipTrigger>
        <TooltipContent>{tooltip}</TooltipContent>
      </Tooltip>
    );
  }

  return button;
};

interface MessageBranchContextType {
  currentBranch: number;
  totalBranches: number;
  goToPrevious: () => void;
  goToNext: () => void;
  branches: ReactElement[];
  setBranches: (branches: ReactElement[]) => void;
}

const MessageBranchContext = createContext<MessageBranchContextType | null>(
  null,
);

const useMessageBranch = () => {
  const context = useContext(MessageBranchContext);

  if (!context) {
    throw new Error("MessageBranch components must be used within");
  }

  return context;
};

export type MessageBranchProps = HTMLAttributes<HTMLDivElement> & {
  defaultBranch?: number;
  onBranchChange?: (branchIndex: number) => void;
};

export const MessageBranch = ({
  defaultBranch = 0,
  onBranchChange,
  className,
  ...props
}: MessageBranchProps) => {
  const [currentBranch, setCurrentBranch] = useState(defaultBranch);
  const [branches, setBranches] = useState<ReactElement[]>([]);

  const handleBranchChange = (newBranch: number) => {
    setCurrentBranch(newBranch);
    onBranchChange?.(newBranch);
  };

  const goToPrevious = () => {
    const newBranch =
      currentBranch > 0 ? currentBranch - 1 : branches.length - 1;
    handleBranchChange(newBranch);
  };

  const goToNext = () => {
    const newBranch =
      currentBranch < branches.length - 1 ? currentBranch + 1 : 0;
    handleBranchChange(newBranch);
  };

  const contextValue: MessageBranchContextType = {
    currentBranch,
    totalBranches: branches.length,
    goToPrevious,
    goToNext,
    branches,
    setBranches,
  };

  return (
    <MessageBranchContext.Provider value={contextValue}>
      <div
        className={cn("grid w-full gap-2 [&>div]:pb-0", className)}
        {...props}
      />
    </MessageBranchContext.Provider>
  );
};

export type MessageBranchContentProps = HTMLAttributes<HTMLDivElement>;

export const MessageBranchContent = ({
  children,
  ...props
}: MessageBranchContentProps) => {
  const { currentBranch, setBranches, branches } = useMessageBranch();
  const childrenArray = Array.isArray(children) ? children : [children];

  // Use useEffect to update branches when they change
  useEffect(() => {
    if (branches.length !== childrenArray.length) {
      setBranches(childrenArray);
    }
  }, [childrenArray, branches, setBranches]);

  return childrenArray.map((branch, index) => (
    <div
      className={cn(
        "grid gap-2 overflow-hidden [&>div]:pb-0",
        index === currentBranch ? "block" : "hidden",
      )}
      key={branch.key}
      {...props}
    >
      {branch}
    </div>
  ));
};

export type MessageBranchSelectorProps = HTMLAttributes<HTMLDivElement> & {
  from: UIMessage["role"];
};

export const MessageBranchSelector = ({
  className,
  from: _from,
  ...props
}: MessageBranchSelectorProps) => {
  const { totalBranches } = useMessageBranch();

  // Don't render if there's only one branch
  if (totalBranches <= 1) {
    return null;
  }

  return (
    <ButtonGroup
      className={cn(
        "[&>*:not(:first-child)]:rounded-l-md [&>*:not(:last-child)]:rounded-r-md",
        className,
      )}
      orientation="horizontal"
      {...props}
    />
  );
};

export type MessageBranchPreviousProps =
  ButtonHTMLAttributes<HTMLButtonElement>;

export const MessageBranchPrevious = ({
  children,
  ...props
}: MessageBranchPreviousProps) => {
  const { goToPrevious, totalBranches } = useMessageBranch();

  return (
    <Button
      aria-label="Previous branch"
      disabled={totalBranches <= 1}
      onClick={goToPrevious}
      size="compact-icon-sm"
      type="button"
      variant="compact-ghost"
      {...props}
    >
      {children ?? <Icon aria-hidden icon={ArrowLeft01Icon} size={14} />}
    </Button>
  );
};

export type MessageBranchNextProps = ButtonHTMLAttributes<HTMLButtonElement>;

export const MessageBranchNext = ({
  children,
  ...props
}: MessageBranchNextProps) => {
  const { goToNext, totalBranches } = useMessageBranch();

  return (
    <Button
      aria-label="Next branch"
      disabled={totalBranches <= 1}
      onClick={goToNext}
      size="compact-icon-sm"
      type="button"
      variant="compact-ghost"
      {...props}
    >
      {children ?? <Icon aria-hidden icon={ArrowRight01Icon} size={14} />}
    </Button>
  );
};

export type MessageBranchPageProps = HTMLAttributes<HTMLSpanElement>;

export const MessageBranchPage = ({
  className,
  ...props
}: MessageBranchPageProps) => {
  const { currentBranch, totalBranches } = useMessageBranch();

  return (
    <div className={cn("flex items-center px-4", className)} {...props}>
      <Text variant="body-medium-tight" tone="muted" as="span">
        {currentBranch + 1} of {totalBranches}
      </Text>
    </div>
  );
};

export type MessageResponseProps = ComponentProps<typeof Streamdown>;

// The renderer carries the markdown, math and code-highlighting libraries, so
// it loads apart from the page; until it is here a reply shows as plain text.
const streamdownResponse = lazyComponent(() =>
  import("./StreamdownResponse").then((module) => module.StreamdownResponse),
);

export const preloadMessageResponse = streamdownResponse.preload;
const StreamdownResponse = streamdownResponse.Loaded;

// memo-boundary: one per text part of every transcript row, and also
// rendered from folders the compiler does not cover (the work output sheet).
export const MessageResponse = memo(function MessageResponse({
  className,
  children,
  ...props
}: MessageResponseProps) {
  const from = useContext(MessageRoleContext);
  if (!streamdownResponse.useIsLoaded()) {
    return (
      <div
        className={cn("size-full break-words whitespace-pre-wrap", className)}
      >
        {children}
      </div>
    );
  }
  return (
    <StreamdownResponse className={className} from={from} {...props}>
      {children}
    </StreamdownResponse>
  );
}, areSameChildren);

function areSameChildren(
  prevProps: MessageResponseProps,
  nextProps: MessageResponseProps,
) {
  return prevProps.children === nextProps.children;
}

MessageResponse.displayName = "MessageResponse";

export type MessageToolbarProps = ComponentProps<"div">;

export const MessageToolbar = ({
  className,
  children,
  ...props
}: MessageToolbarProps) => (
  <div
    className={cn(
      "mt-4 flex w-full items-center justify-between gap-4",
      className,
    )}
    {...props}
  >
    {children}
  </div>
);
