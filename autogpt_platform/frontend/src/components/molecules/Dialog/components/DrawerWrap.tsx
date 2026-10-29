import { scrollbarStyles } from "@/components/styles/scrollbars";
import { isComposingEvent } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { PropsWithChildren } from "react";
import { Drawer } from "vaul";
import { DialogCtx } from "../useDialogCtx";
import { compactStyles, drawerStyles, modalStyles } from "./styles";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type BaseProps = DialogCtx & PropsWithChildren;

interface Props extends BaseProps {
  testId?: string;
  title: React.ReactNode;
  handleClose: () => void;
}

export function DrawerWrap({
  children,
  title,
  variant,
  testId,
  handleClose,
  isForceOpen,
  className,
}: Props) {
  const accessibleTitle = title || "Dialog";
  const hasVisibleTitle = Boolean(title);
  const isCompact = variant === "compact";

  // Mirrors DialogWrap: below the lg breakpoint the same <Dialog> renders as a
  // drawer, and Escape has to behave identically in both.
  function handleEscapeKeyDown(event: KeyboardEvent) {
    if (isForceOpen || isComposingEvent(event)) event.preventDefault();
  }

  const closeBtn = (
    <button
      type="button"
      aria-label="Close"
      onClick={handleClose}
      data-slot="drawer-close"
      className="inline-flex h-9 items-center justify-center rounded-md text-foreground-strong transition-colors focus-visible:outline-none"
    >
      <Icon
        data-slot="drawer-close-icon"
        icon={Cancel01Icon}
        width={isCompact ? "1.25rem" : "1.5rem"}
      />
    </button>
  );

  return (
    <Drawer.Portal>
      <Drawer.Overlay
        data-slot="drawer-overlay"
        className={drawerStyles.overlay}
      />
      <Drawer.Content
        data-slot="drawer-popup"
        className={cn(
          drawerStyles.content,
          isCompact && compactStyles.drawerContent,
          className,
        )}
        data-testid={testId}
        onEscapeKeyDown={handleEscapeKeyDown}
        // No onInteractOutside close: Radix dismisses outside taps itself and
        // vetoes the focus a closing DropdownMenu hands back to its trigger.
      >
        <div
          data-slot="drawer-header"
          className={cn(
            "flex w-full shrink-0 items-center justify-between",
            hasVisibleTitle
              ? isCompact
                ? compactStyles.header
                : "pb-6"
              : "pb-0",
          )}
        >
          {hasVisibleTitle ? (
            <Drawer.Title
              data-slot="drawer-title"
              className={isCompact ? compactStyles.title : drawerStyles.title}
            >
              {accessibleTitle}
            </Drawer.Title>
          ) : (
            <Drawer.Title data-slot="drawer-title" className="sr-only">
              {accessibleTitle}
            </Drawer.Title>
          )}

          {!isForceOpen ? (
            hasVisibleTitle ? (
              closeBtn
            ) : (
              <div
                data-slot="drawer-close-wrap"
                className={`${modalStyles.iconWrap} transition-colors duration-200 hover:bg-surface-active`}
              >
                {closeBtn}
              </div>
            )
          ) : null}
        </div>
        <Drawer.Description data-slot="drawer-description" className="sr-only">
          Dialog
        </Drawer.Description>
        <div
          data-slot="drawer-body"
          className="flex min-h-0 flex-1 flex-col overflow-hidden"
        >
          <div
            data-slot="drawer-viewport"
            className={cn(
              "flex-1 overflow-y-auto overflow-x-hidden",
              scrollbarStyles,
            )}
          >
            {children}
          </div>
        </div>
      </Drawer.Content>
    </Drawer.Portal>
  );
}
