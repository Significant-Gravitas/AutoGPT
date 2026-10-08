"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { isComposingEscape } from "@/components/molecules/Dialog/helpers";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import {
  Sheet as KobraSheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { forwardRef, isValidElement, ReactNode } from "react";
import { panelClassName, SheetSide } from "./helpers";

interface Props {
  /** Accessible name of the sheet. Shown in the header unless `hideTitle`. */
  title: ReactNode;
  hideTitle?: boolean;
  description?: ReactNode;
  hideDescription?: boolean;
  /** Element that opens the sheet; rendered as the trigger itself. */
  trigger?: ReactNode;
  /** Header controls shown next to the close button. */
  actions?: ReactNode;
  footer?: ReactNode;
  side?: SheetSide | null;
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
  const resolvedSide = side ?? "right";
  const hasVisibleHeader =
    !hideTitle || Boolean(description && !hideDescription);

  return (
    <KobraSheet
      open={open}
      defaultOpen={defaultOpen}
      onOpenChange={(next, details) => {
        // Escape dismisses an IME candidate window, not the sheet; see
        // AGENTS.md "Keyboard handling".
        if (!next && isComposingEscape(details)) {
          details.cancel();
          return;
        }
        onOpenChange?.(next);
      }}
    >
      {isValidElement(trigger) ? (
        <SheetTrigger render={trigger} />
      ) : trigger ? (
        <SheetTrigger>{trigger}</SheetTrigger>
      ) : null}
      <SheetContent
        ref={ref}
        side={resolvedSide}
        showCloseButton={false}
        className={cn(panelClassName(resolvedSide), className)}
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
            <SheetTitle
              render={
                <Text
                  variant="large-semibold"
                  as="h2"
                  tone="primary"
                  className={cn("wrap-break-word", hideTitle && "sr-only")}
                />
              }
            >
              {title}
            </SheetTitle>
            {description ? (
              <SheetDescription
                render={
                  <Text
                    variant="body"
                    tone="secondary"
                    className={cn(hideDescription && "sr-only")}
                  />
                }
              >
                {description}
              </SheetDescription>
            ) : null}
          </div>
          <div className="ml-auto flex shrink-0 items-center gap-2">
            {actions}
            <SheetClose
              render={
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Close"
                  withTooltip={false}
                >
                  <Icon icon={Cancel01Icon} size={16} aria-hidden />
                </Button>
              }
            />
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
          <div className="flex shrink-0 justify-end gap-2 border-t border-border px-6 py-4">
            {footer}
          </div>
        ) : null}
      </SheetContent>
    </KobraSheet>
  );
});
