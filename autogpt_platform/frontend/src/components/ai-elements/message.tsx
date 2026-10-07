"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Button } from "@/components/atoms/Button/Button";
import type { ButtonProps } from "@/components/atoms/Button/helpers";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import { cjk } from "@streamdown/cjk";
import { code } from "@/lib/streamdown-code-plugin";
import { createMathPlugin } from "@streamdown/math";
import { escapeCurrencyAmounts } from "@/lib/markdown-math";
import { mermaid } from "@streamdown/mermaid";
import type { UIMessage } from "ai";
import type { ComponentProps, HTMLAttributes } from "react";
import { memo } from "react";
import type { LinkSafetyModalProps } from "streamdown";
import { Streamdown } from "streamdown";

export type MessageProps = HTMLAttributes<HTMLDivElement> & {
  from: UIMessage["role"];
};

export const Message = ({ className, from, ...props }: MessageProps) => (
  <div
    className={cn(
      "group flex w-full max-w-[95%] flex-col gap-2",
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
      "is-user:dark flex w-full min-w-0 max-w-full flex-col gap-2 overflow-hidden text-sm",
      "group-[.is-user]:w-fit",
      "group-[.is-user]:ml-auto group-[.is-user]:rounded-lg group-[.is-user]:bg-zinc-100 group-[.is-user]:px-4 group-[.is-user]:py-3 group-[.is-user]:text-zinc-950",
      "group-[.is-assistant]:text-zinc-950",
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

type ButtonElementProps = Extract<ButtonProps, { as?: "button" }>;

export type MessageActionProps = ButtonElementProps & {
  tooltip?: string;
  label?: string;
};

export const MessageAction = ({
  tooltip,
  children,
  label,
  variant = "ghost",
  size = "icon-sm",
  ...props
}: MessageActionProps) => {
  const button = (
    <Button size={size} type="button" variant={variant} {...props}>
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
          variant="small"
          className="mt-2 break-all rounded-md bg-zinc-100 p-3 font-mono"
          unmask={false}
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
const math = createMathPlugin({ singleDollarTextMath: true });

export const MessageResponse = memo(
  ({ className, children, components, ...props }: MessageResponseProps) => (
    <Streamdown
      className={cn(
        "size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&_pre]:!bg-white",
        "[&_a]:text-blue-500 [&_a]:no-underline hover:[&_a]:underline",
        // Raycast/Linear-style markdown tables — clean borders, subtle row
        // separators, light header, no outer-corner artifacts.
        "[&_table]:w-full [&_table]:border-separate [&_table]:border-spacing-0 [&_table]:border [&_table]:border-zinc-200 [&_table]:text-sm",
        "[&_thead]:bg-zinc-50",
        "[&_th]:border-b [&_th]:border-zinc-200 [&_th]:px-3 [&_th]:py-2 [&_th]:text-left [&_th]:font-medium [&_th]:text-zinc-600",
        "[&_td]:border-b [&_td]:border-zinc-100 [&_td]:px-3 [&_td]:py-2 [&_td]:align-top [&_td]:text-zinc-800",
        "[&_tbody_tr:last-child_td]:border-b-0",
        "[&_tbody_tr:hover_td]:bg-zinc-50/60",
        className,
      )}
      components={{
        // Tables can be wider than the chat column (long URLs, many columns).
        // Wrap in an overflow-x-auto div so the page never grows past the
        // chat container — the table scrolls internally instead.
        table: ({ children, ...tableProps }) => (
          <div className="my-4 max-w-full overflow-x-auto">
            <table {...tableProps}>{children}</table>
          </div>
        ),
        ...components,
      }}
      plugins={{ code, mermaid, math, cjk }}
      linkSafety={{
        enabled: true,
        onLinkCheck: isSameOriginLink,
        renderModal: (modalProps) => <ExternalLinkModal {...modalProps} />,
      }}
      {...props}
    >
      {typeof children === "string"
        ? escapeCurrencyAmounts(children)
        : children}
    </Streamdown>
  ),
  (prevProps, nextProps) => prevProps.children === nextProps.children,
);

MessageResponse.displayName = "MessageResponse";
