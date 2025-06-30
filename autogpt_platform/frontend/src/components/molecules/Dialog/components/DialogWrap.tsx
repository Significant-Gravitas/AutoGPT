import { cn } from "@/lib/utils";
import { X } from "@phosphor-icons/react";
import * as RXDialog from "@radix-ui/react-dialog";
import { CSSProperties, PropsWithChildren } from "react";
import { DialogCtx } from "../useDialogCtx";
import { modalStyles } from "./styles";

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
            <button
              type="button"
              onClick={handleClose}
              aria-label="Close"
              className={`${modalStyles.iconWrap} transition-colors duration-200 hover:bg-gray-200 dark:hover:bg-stone-900`}
            >
              <X className={modalStyles.icon} />
            </button>
          )}
        </div>
        <div className="overflow-y-auto">{children}</div>
      </RXDialog.Content>
    </RXDialog.Portal>
  );
}
