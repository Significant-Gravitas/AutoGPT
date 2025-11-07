import { cn } from "@/lib/utils";
import { X } from "@phosphor-icons/react";
import * as RXDialog from "@radix-ui/react-dialog";
import {
  CSSProperties,
  PropsWithChildren,
  useEffect,
  useRef,
  useState,
} from "react";
import { DialogCtx } from "../useDialogCtx";
import { modalStyles } from "./styles";
import { scrollbarStyles } from "@/components/styles/scrollbars";

type BaseProps = DialogCtx & PropsWithChildren;

interface Props extends BaseProps {
  title: React.ReactNode;
  styling: CSSProperties | undefined;
  withGradient?: boolean;
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
      <RXDialog.Overlay className={modalStyles.overlay} />
      <RXDialog.Content
        onInteractOutside={handleClose}
        onEscapeKeyDown={handleClose}
        aria-describedby={undefined}
        className={cn(modalStyles.content)}
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
            <button
              onClick={handleClose}
              aria-label="Close"
              className="absolute right-4 top-4 z-50 hover:border-transparent hover:bg-transparent focus:border-none focus:outline-none"
            >
              <X className={modalStyles.icon} />
            </button>
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
