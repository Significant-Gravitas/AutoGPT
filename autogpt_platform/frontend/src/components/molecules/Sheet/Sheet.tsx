"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { isComposingEvent } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import * as RXDialog from "@radix-ui/react-dialog";
import { type VariantProps } from "class-variance-authority";
import { forwardRef, ReactNode } from "react";
import { overlayClassName, sheetVariants } from "./helpers";

interface Props extends VariantProps<typeof sheetVariants> {
  /** Accessible name of the sheet. Shown in the header unless `hideTitle`. */
  title: ReactNode;
  hideTitle?: boolean;
  description?: ReactNode;
  hideDescription?: boolean;
  /** Element that opens the sheet, rendered through Radix `Trigger asChild`. */
  trigger?: ReactNode;
  /** Header controls shown next to the close button. */
  actions?: ReactNode;
  footer?: ReactNode;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** Classes for the panel, e.g. a wider `sm:max-w-xl`. */
  className?: string;
  bodyClassName?: string;
  children?: ReactNode;
}

export const Sheet = forwardRef<HTMLDivElement, Props>(function Sheet(
  {
    title,
    hideTitle = false,
    description,
    hideDescription = false,
    trigger,
    actions,
    footer,
    side,
    open,
    defaultOpen,
    onOpenChange,
    className,
    bodyClassName,
    children,
  },
  ref,
) {
  // Escape dismisses an IME candidate window, not the sheet; see AGENTS.md
  // "Keyboard handling".
  function handleEscapeKeyDown(event: KeyboardEvent) {
    if (isComposingEvent(event)) event.preventDefault();
  }

  const hasVisibleHeader =
    !hideTitle || Boolean(description && !hideDescription);

  return (
    <RXDialog.Root
      open={open}
      defaultOpen={defaultOpen}
      onOpenChange={onOpenChange}
    >
      {trigger ? <RXDialog.Trigger asChild>{trigger}</RXDialog.Trigger> : null}
      <RXDialog.Portal>
        <RXDialog.Overlay className={overlayClassName} />
        <RXDialog.Content
          ref={ref}
          onEscapeKeyDown={handleEscapeKeyDown}
          // Without a description, opt out of Radix's missing-description
          // warning; with one, keep the link Radix sets up.
          {...(description ? {} : { "aria-describedby": undefined })}
          className={cn(sheetVariants({ side }), className)}
        >
          <div
            className={cn(
              "flex shrink-0 items-start gap-2 p-6",
              hasVisibleHeader ? "pb-4" : "pb-0",
            )}
          >
            <div
              className={cn(
                "flex min-w-0 flex-1 flex-col gap-1",
                !hasVisibleHeader && "sr-only",
              )}
            >
              <RXDialog.Title asChild>
                <Text
                  variant="large-semibold"
                  as="h2"
                  tone="primary"
                  className={cn("wrap-break-word", hideTitle && "sr-only")}
                >
                  {title}
                </Text>
              </RXDialog.Title>
              {description ? (
                <RXDialog.Description asChild>
                  <Text
                    variant="body"
                    tone="secondary"
                    className={cn(hideDescription && "sr-only")}
                  >
                    {description}
                  </Text>
                </RXDialog.Description>
              ) : null}
            </div>
            <div className="ml-auto flex shrink-0 items-center gap-2">
              {actions}
              <RXDialog.Close asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Close"
                  withTooltip={false}
                  className="focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2"
                >
                  <Icon icon={Cancel01Icon} size={16} aria-hidden />
                </Button>
              </RXDialog.Close>
            </div>
          </div>
          <div
            className={cn(
              "flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-6 pb-6",
              scrollbarStyles,
              bodyClassName,
            )}
          >
            {children}
          </div>
          {footer ? (
            <div className="flex shrink-0 justify-end gap-2 border-t border-zinc-200 px-6 py-4">
              {footer}
            </div>
          ) : null}
        </RXDialog.Content>
      </RXDialog.Portal>
    </RXDialog.Root>
  );
});
