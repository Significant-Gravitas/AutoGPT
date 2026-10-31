"use client";

import { ButtonGroup, ButtonGroupText } from "./components/ButtonGroup";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { MapAwarePre } from "@/components/contextual/MapEmbed/MapAwarePre";
import { cn } from "@/lib/utils";
import { cjk } from "@streamdown/cjk";
import { code } from "@/lib/streamdown-code-plugin";
import { createMathPlugin, type MathPlugin } from "@streamdown/math";
import { escapeCurrencyAmounts } from "@/lib/markdown-math";
import { mermaid } from "@/lib/streamdown-mermaid-plugin";
import rehypeKatex from "rehype-katex";
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
import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import type { LinkSafetyModalProps } from "streamdown";
import { defaultRemarkPlugins, Streamdown } from "streamdown";
import { remarkAppLinks } from "./app-links";
import {
  createComponentsMerger,
  type StreamdownComponents,
} from "./streamdown-components";

export type MessageProps = HTMLAttributes<HTMLDivElement> & {
  from: UIMessage["role"];
};

export const Message = ({ className, from, ...props }: MessageProps) => (
  <div
    className={cn(
      "group flex w-full max-w-[95%] flex-col gap-2",
      // oxlint-disable-next-line design-system/no-unknown-classes -- is-user and is-assistant are markers the group-[.is-*] variants below match
      from === "user" ? "is-user ml-auto justify-end" : "is-assistant",
      className,
    )}
    {...props}
  />
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
    <ButtonGroupText
      className={cn(
        "border-none bg-transparent text-foreground-tertiary shadow-none",
        className,
      )}
      {...props}
    >
      {currentBranch + 1} of {totalBranches}
    </ButtonGroupText>
  );
};

export type MessageResponseProps = ComponentProps<typeof Streamdown>;

function isSameOriginLink(url: string): boolean {
  try {
    const parsed = new URL(url, window.location.origin);
    return parsed.origin === window.location.origin;
  } catch {
    return false;
  }
}

function ExternalLinkModal({
  url,
  isOpen,
  onClose,
  onConfirm,
}: LinkSafetyModalProps) {
  return (
    <Dialog
      title="Open external link"
      styling={{ maxWidth: "30rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body">
          You&apos;re about to visit an external website:
        </Text>
        <Text
          variant="small-mono"
          className="mt-2 rounded-md bg-surface-hover p-3 break-all"
        >
          {url}
        </Text>
        <Dialog.Footer>
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" onClick={onConfirm}>
            Open link
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

// Single-dollar math is off by default in @streamdown/math, so "$x^2$" reached the
// reader raw; escapeCurrencyAmounts keeps prices out of the formulas it enables.
// KaTeX renders text real LaTeX would reject, such as Bengali in a formula, but
// its default strict mode console.warns each time, and console capture filed
// every warning as a Sentry issue (BUILDER-94S). The plugin has no option for
// strict mode, so its rehype step is rebuilt with it off.
const math: MathPlugin = {
  ...createMathPlugin({ singleDollarTextMath: true }),
  rehypePlugin: [
    rehypeKatex,
    { errorColor: "var(--color-muted-foreground)", strict: "ignore" },
  ],
};

// Streamdown re-parses every block of a message when the identity of a
// component in `components` changes, and React remounts any element whose
// component type changes, so everything handed to Streamdown is built once.
function MessageTable({ children, ...tableProps }: ComponentProps<"table">) {
  // Tables can be wider than the chat column (long URLs, many columns).
  // Wrap in an overflow-x-auto div so the page never grows past the
  // chat container — the table scrolls internally instead.
  return (
    <div className="my-4 max-w-full overflow-x-auto rounded-xl border border-border-base bg-surface shadow-message-table">
      <table {...tableProps}>{children}</table>
    </div>
  );
}

function renderExternalLinkModal(modalProps: LinkSafetyModalProps) {
  return <ExternalLinkModal {...modalProps} />;
}

const STREAMDOWN_PLUGINS = { code, mermaid, math, cjk };

// Streamdown memoises every block on the identity of this list, so it is built
// once here.
const REMARK_PLUGINS = [...Object.values(defaultRemarkPlugins), remarkAppLinks];

const LINK_SAFETY = {
  enabled: true,
  onLinkCheck: isSameOriginLink,
  renderModal: renderExternalLinkModal,
};

const TABLE_COMPONENT: StreamdownComponents = {
  table: MessageTable,
  pre: MapAwarePre,
};

const mergeTableComponent = createComponentsMerger((components) => ({
  ...TABLE_COMPONENT,
  ...components,
}));

function withTableComponent(components: StreamdownComponents | undefined) {
  return components ? mergeTableComponent(components) : TABLE_COMPONENT;
}

// memo-boundary: one per text part of every transcript row, and also
// rendered from folders the compiler does not cover (the work output sheet).
export const MessageResponse = memo(function MessageResponse({
  className,
  children,
  components,
  ...props
}: MessageResponseProps) {
  // Merged on every render, as a plain call would be: the merger itself keeps
  // the result stable, and spots an entry the caller replaced in place, which
  // a result cached on `components` would miss.
  const mergedComponents = useReadEveryRender(() =>
    withTableComponent(components),
  );
  return (
    <Streamdown
      className={cn(
        "size-full [&_pre]:!bg-surface [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        "[&_a]:text-info [&_a]:no-underline [&_a]:hover:underline",
        // Raycast-style markdown tables: a rounded card with a quiet small-caps
        // header, hairline row separators and tabular figures.
        "[&_table]:w-full [&_table]:border-separate [&_table]:border-spacing-0 [&_table]:border-0 [&_table]:text-body-sm [&_table]:leading-inherit [&_table]:tabular-nums",
        "[&_tbody]:bg-transparent [&_thead]:bg-message-table-header",
        // Streamdown merges className again with its own tailwind-merge
        // (default config), which reads the size roles as colours and drops
        // them beside a colour in the same scope; the sizes therefore sit in
        // their own [&_table_*] scope. th and inline code keep the 20px
        // line-height their own text-sm gives them.
        "[&_table_td]:text-body-sm [&_table_td_code]:text-caption [&_table_td_code]:leading-5 [&_table_th]:text-caption-sm [&_table_th]:leading-5",
        "[&_th]:h-9 [&_th]:border-b [&_th]:border-border-base [&_th]:px-4 [&_th]:py-0 [&_th]:text-left [&_th]:font-medium [&_th]:tracking-eyebrow [&_th]:whitespace-nowrap [&_th]:uppercase",
        "[&_th]:text-shadow-message-table-indigo [&_th:first-child]:text-shadow-message-table-violet [&_th:last-child]:text-shadow-message-table-blue",
        "[&_td]:border-b [&_td]:border-separator [&_td]:px-4 [&_td]:py-2 [&_td]:align-top [&_td]:text-foreground-body [&_td]:transition-colors",
        "[&_td_strong]:font-medium [&_td_strong]:text-foreground-base",
        "[&_td_code]:rounded [&_td_code]:bg-surface-hover [&_td_code]:px-1 [&_td_code]:py-px [&_td_code]:whitespace-nowrap [&_td_code]:text-foreground-body",
        "[&_tbody_tr:last-child_td]:border-b-0",
        "[&_tbody_tr:hover_td]:bg-surface-muted",
        className,
      )}
      components={mergedComponents}
      plugins={STREAMDOWN_PLUGINS}
      remarkPlugins={REMARK_PLUGINS}
      linkSafety={LINK_SAFETY}
      {...props}
    >
      {typeof children === "string"
        ? escapeCurrencyAmounts(children)
        : children}
    </Streamdown>
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
