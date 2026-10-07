import { Button } from "@/components/atoms/Button/Button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { isComposingEvent } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import * as RXDialog from "@radix-ui/react-dialog";
import {
  CSSProperties,
  PropsWithChildren,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { DialogCtx, DialogVariant } from "../useDialogCtx";
import { compactStyles, modalStyles } from "./styles";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

type BaseProps = DialogCtx & PropsWithChildren;

interface Props extends BaseProps {
  title: React.ReactNode;
  variant: DialogVariant;
  styling: CSSProperties | undefined;
  withGradient?: boolean;
}

/**
 * Check if an external picker (like Google Drive) is currently open.
 * Used to prevent dialog from closing when user interacts with the picker.
 */
function isExternalPickerOpen(): boolean {
  return document.body.hasAttribute("data-google-picker-open");
}

export function DialogWrap({
  children,
  title,
  description,
  hideDescription,
  variant,
  styling = {},
  className,
  isForceOpen,
  handleClose,
}: Props) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [hasVerticalScrollbar, setHasVerticalScrollbar] = useState(false);
  const isCompact = variant === "compact";
  const hasVisibleHeader = Boolean(title || (description && !hideDescription));

  // Prevent dialog from closing when external picker is open or when forceOpen is true
  const handleInteractOutside = useCallback(
    (event: Event) => {
      if (isExternalPickerOpen() || isForceOpen) {
        event.preventDefault();
        return;
      }
      handleClose();
    },
    [handleClose, isForceOpen],
  );

  const handlePointerDownOutside = useCallback(
    (event: Event) => {
      if (isExternalPickerOpen() || isForceOpen) {
        event.preventDefault();
      }
    },
    [isForceOpen],
  );

  const handleFocusOutside = useCallback(
    (event: Event) => {
      if (isExternalPickerOpen() || isForceOpen) {
        event.preventDefault();
      }
    },
    [isForceOpen],
  );

  // Radix closes the dialog on Escape by default, through onOpenChange. Veto it
  // while an IME is composing, so Escape dismisses the candidate window without
  // losing the dialog, and when the dialog is force-open — matching the sibling
  // handlers above and skipping a dismiss path that Dialog.tsx would undo.
  function handleEscapeKeyDown(event: KeyboardEvent) {
    if (isForceOpen || isComposingEvent(event)) event.preventDefault();
  }

  useEffect(() => {
    function update() {
      const el = scrollRef.current;
      if (!el) return;
      setHasVerticalScrollbar(el.scrollHeight > el.clientHeight + 1);
    }
    update();
    const ro = new ResizeObserver(update);
    if (scrollRef.current) ro.observe(scrollRef.current);
    window.addEventListener("resize", update);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", update);
    };
  }, []);

  return (
    <RXDialog.Portal>
      <RXDialog.Overlay data-dialog-overlay className={modalStyles.overlay} />
      <RXDialog.Content
        data-dialog-content
        onInteractOutside={handleInteractOutside}
        onPointerDownOutside={handlePointerDownOutside}
        onFocusOutside={handleFocusOutside}
        onEscapeKeyDown={handleEscapeKeyDown}
        // Without a description, opt out of Radix's missing-description
        // warning; with one, keep the link Radix sets up.
        {...(description ? {} : { "aria-describedby": undefined })}
        className={cn(
          modalStyles.content,
          isCompact && compactStyles.content,
          className,
        )}
        style={{
          ...styling,
        }}
      >
        <div
          className={cn(
            "flex items-center justify-between px-2",
            hasVisibleHeader
              ? isCompact
                ? compactStyles.header
                : "pb-6"
              : "pb-0",
          )}
        >
          <div className="flex min-w-0 flex-col gap-2">
            {title ? (
              <RXDialog.Title
                className={isCompact ? compactStyles.title : modalStyles.title}
              >
                {title}
              </RXDialog.Title>
            ) : (
              <RXDialog.Title className="sr-only">Dialog</RXDialog.Title>
            )}
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

          {isForceOpen ? null : (
            <Button
              variant="icon"
              size={isCompact ? "icon-sm" : "icon-lg"}
              onClick={handleClose}
              aria-label="Close"
              className={cn(
                "absolute top-4 right-4 z-50 bg-popover",
                isCompact ? compactStyles.close : "size-10",
              )}
              withTooltip={false}
            >
              <Icon icon={Cancel01Icon} width="1rem" />
            </Button>
          )}
        </div>
        <div className="flex min-h-0 flex-1 flex-col">
          <div
            ref={scrollRef}
            className={cn(
              "flex-1 overflow-x-hidden overflow-y-auto px-2",
              scrollbarStyles,
              hasVerticalScrollbar
                ? "-mr-6 scrollbar-gutter-stable"
                : "mr-0 scrollbar-gutter-auto",
            )}
          >
            {children}
          </div>
        </div>
      </RXDialog.Content>
    </RXDialog.Portal>
  );
}
