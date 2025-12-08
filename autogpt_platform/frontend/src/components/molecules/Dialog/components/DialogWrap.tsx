import { Button } from "@/components/atoms/Button/Button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { X } from "@phosphor-icons/react";
import * as RXDialog from "@radix-ui/react-dialog";
import {
  CSSProperties,
  PropsWithChildren,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { DialogCtx } from "../useDialogCtx";
import { modalStyles } from "./styles";

type BaseProps = DialogCtx & PropsWithChildren;

interface Props extends BaseProps {
  title: React.ReactNode;
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
  styling = {},
  isForceOpen,
  handleClose,
}: Props) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [hasVerticalScrollbar, setHasVerticalScrollbar] = useState(false);

  // Prevent dialog from closing when external picker is open
  const handleInteractOutside = useCallback(
    (event: Event) => {
      if (isExternalPickerOpen()) {
        event.preventDefault();
        return;
      }
      handleClose();
    },
    [handleClose],
  );

  const handlePointerDownOutside = useCallback((event: Event) => {
    if (isExternalPickerOpen()) {
      event.preventDefault();
    }
  }, []);

  const handleFocusOutside = useCallback((event: Event) => {
    if (isExternalPickerOpen()) {
      event.preventDefault();
    }
  }, []);

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
        onEscapeKeyDown={handleClose}
        aria-describedby={undefined}
        className={modalStyles.content}
        style={{
          ...styling,
        }}
      >
        <div
          className={`flex items-center justify-between ${
            title ? "pb-6" : "pb-0"
          }`}
        >
          {title ? (
            <RXDialog.Title className={modalStyles.title}>
              {title}
            </RXDialog.Title>
          ) : (
            <span className="sr-only">
              {/* Title is required for a11y compliance even if not displayed so screen readers can announce it */}
              <RXDialog.Title>{title}</RXDialog.Title>
            </span>
          )}

          {isForceOpen && !handleClose ? null : (
            <Button
              variant="icon"
              size="icon"
              onClick={handleClose}
              aria-label="Close"
              className="absolute right-4 top-4 z-50 size-[2.5rem] bg-white"
              withTooltip={false}
            >
              <X width="1rem" />
            </Button>
          )}
        </div>
        <div className="flex min-h-0 flex-1 flex-col">
          <div
            ref={scrollRef}
            className={cn(
              "flex-1 overflow-y-auto overflow-x-hidden",
              scrollbarStyles,
            )}
            style={{
              scrollbarGutter: "stable",
              marginRight: hasVerticalScrollbar ? "-14px" : "0px",
            }}
          >
            {children}
          </div>
        </div>
      </RXDialog.Content>
    </RXDialog.Portal>
  );
}
