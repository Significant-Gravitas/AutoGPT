import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import {
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import {
  CSSProperties,
  PropsWithChildren,
  useEffect,
  useRef,
  useState,
} from "react";
import { DialogCtx, DialogVariant } from "../useDialogCtx";
import { compactStyles, modalStyles } from "./styles";

type BaseProps = DialogCtx & PropsWithChildren;

interface Props extends BaseProps {
  title: React.ReactNode;
  variant: DialogVariant;
  styling: CSSProperties | undefined;
  withGradient?: boolean;
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
    <DialogContent
      data-dialog-content
      showCloseButton={false}
      className={cn(
        modalStyles.content,
        isCompact && compactStyles.content,
        className,
      )}
      style={styling}
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
            <DialogTitle
              className={isCompact ? compactStyles.title : modalStyles.title}
            >
              {title}
            </DialogTitle>
          ) : (
            <DialogTitle className="sr-only">Dialog</DialogTitle>
          )}
          {description ? (
            <DialogDescription
              render={
                <Text
                  variant="body"
                  tone="secondary"
                  className={cn(hideDescription && "sr-only")}
                />
              }
            >
              {description}
            </DialogDescription>
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
    </DialogContent>
  );
}
