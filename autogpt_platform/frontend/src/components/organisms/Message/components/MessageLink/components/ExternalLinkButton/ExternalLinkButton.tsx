import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { ExternalLinkModal } from "../ExternalLinkModal/ExternalLinkModal";
import { useExternalLinkButton } from "./useExternalLinkButton";

interface Props {
  url: string;
  title?: string;
  className?: string;
  children?: ReactNode;
}

// A button, not an anchor, so no click and no "Open in new tab" menu can leave
// the site without the dialog. Streamdown's own link look; globals.css wraps it.
export function ExternalLinkButton({ url, title, className, children }: Props) {
  const { isAsking, handleClick, handleMouseDown, handleConfirm, handleClose } =
    useExternalLinkButton(url);

  return (
    <>
      <button
        type="button"
        title={title}
        aria-haspopup="dialog"
        onClick={handleClick}
        onAuxClick={handleClick}
        onMouseDown={handleMouseDown}
        data-streamdown="link"
        className={cn(
          "appearance-none text-left font-medium text-action underline",
          className,
        )}
      >
        {children}
      </button>
      <ExternalLinkModal
        url={url}
        isOpen={isAsking}
        onClose={handleClose}
        onConfirm={handleConfirm}
      />
    </>
  );
}
