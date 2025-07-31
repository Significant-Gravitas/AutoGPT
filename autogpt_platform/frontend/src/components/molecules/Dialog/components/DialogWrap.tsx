import { cn } from "@/lib/utils";
import { X } from "@phosphor-icons/react";
import * as RXDialog from "@radix-ui/react-dialog";
import { CSSProperties, PropsWithChildren } from "react";
import { DialogCtx } from "../useDialogCtx";
import { modalStyles } from "./styles";
import styles from "./styles.module.css";
import { Button } from "@/components/atoms/Button/Button";

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
            <Button
              variant="ghost"
              onClick={handleClose}
              aria-label="Close"
              className="absolute -right-2 top-2 z-50 hover:border-transparent hover:bg-transparent"
            >
              <X className={modalStyles.icon} />
            </Button>
          )}
        </div>
        <div className={`overflow-y-auto ${styles.scrollableContent}`}>
          {children}
        </div>
      </RXDialog.Content>
    </RXDialog.Portal>
  );
}
