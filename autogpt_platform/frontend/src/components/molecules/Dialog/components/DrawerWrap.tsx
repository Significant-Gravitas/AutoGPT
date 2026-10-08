import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import {
  DrawerContent,
  DrawerDescription,
  DrawerTitle,
} from "@/components/ui/drawer";
import { isComposingEvent } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { PropsWithChildren } from "react";
import { DialogCtx } from "../useDialogCtx";
import { compactStyles, drawerStyles } from "./styles";

type BaseProps = DialogCtx & PropsWithChildren;

interface Props extends BaseProps {
  testId?: string;
  title: React.ReactNode;
  handleClose: () => void;
}

export function DrawerWrap({
  children,
  title,
  description,
  hideDescription,
  variant,
  testId,
  handleClose,
  isForceOpen,
  className,
}: Props) {
  const accessibleTitle = title || "Dialog";
  const hasVisibleTitle = Boolean(title);
  const isCompact = variant === "compact";
  const hasVisibleHeader =
    hasVisibleTitle || Boolean(description && !hideDescription);

  // Mirrors DialogWrap: below the lg breakpoint the same <Dialog> renders as a
  // drawer, and Escape has to behave identically in both.
  function handleEscapeKeyDown(event: KeyboardEvent) {
    if (isForceOpen || isComposingEvent(event)) event.preventDefault();
  }

  return (
    <DrawerContent
      className={cn(
        drawerStyles.content,
        isCompact && compactStyles.drawerContent,
        className,
      )}
      data-testid={testId}
      onEscapeKeyDown={handleEscapeKeyDown}
      {...(description ? {} : { "aria-describedby": undefined })}
      // No onInteractOutside close: vaul dismisses outside taps itself and
      // vetoes the focus a closing DropdownMenu hands back to its trigger.
    >
      <div
        className={cn(
          "flex w-full shrink-0 items-center justify-between",
          hasVisibleHeader
            ? isCompact
              ? compactStyles.header
              : "pb-6"
            : "pb-0",
        )}
      >
        <div className="flex min-w-0 flex-col gap-2">
          {hasVisibleTitle ? (
            <DrawerTitle
              className={isCompact ? compactStyles.title : drawerStyles.title}
            >
              {accessibleTitle}
            </DrawerTitle>
          ) : (
            <DrawerTitle className="sr-only">{accessibleTitle}</DrawerTitle>
          )}
          {description ? (
            <DrawerDescription asChild>
              <Text
                variant="body"
                tone="secondary"
                className={cn(hideDescription && "sr-only")}
              >
                {description}
              </Text>
            </DrawerDescription>
          ) : null}
        </div>

        {isForceOpen ? null : (
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Close"
            onClick={handleClose}
            className="focus-visible:ring-0!"
            withTooltip={false}
          >
            <Icon
              icon={Cancel01Icon}
              width={isCompact ? "1.25rem" : "1.5rem"}
            />
          </Button>
        )}
      </div>
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div
          className={cn(
            "flex-1 overflow-x-hidden overflow-y-auto",
            scrollbarStyles,
          )}
        >
          {children}
        </div>
      </div>
    </DrawerContent>
  );
}
