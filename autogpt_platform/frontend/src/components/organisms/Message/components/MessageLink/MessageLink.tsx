import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";
import { ExternalLinkModal } from "./components/ExternalLinkModal";
import { INCOMPLETE_LINK_HREF } from "./helpers";
import { useMessageLink } from "./useMessageLink";

interface Props extends ComponentProps<"a"> {
  node?: unknown;
}

// Streamdown's own link turns into a button without an href while link safety
// is on, so middle-click, Ctrl/Cmd-click and "Open in new tab" had nothing to
// open. This one stays a real anchor and asks only on the clicks helpers names.
export function MessageLink({
  children,
  className,
  href,
  node: _node,
  ...props
}: Props) {
  const isIncomplete = href === INCOMPLETE_LINK_HREF;
  const url = isIncomplete ? undefined : href;
  const { isAsking, handleClick, handleConfirm, handleClose } =
    useMessageLink(url);

  return (
    <>
      <a
        {...props}
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        onClick={handleClick}
        onAuxClick={handleClick}
        data-incomplete={isIncomplete}
        data-streamdown="link"
        // Doubled to outrank the chat's `[&_a]` colour, as WorkspaceLink does.
        className={cn(
          "font-medium [&&]:text-primary [&&]:underline",
          className,
        )}
      >
        {children}
      </a>
      {url ? (
        <ExternalLinkModal
          url={url}
          isOpen={isAsking}
          onClose={handleClose}
          onConfirm={handleConfirm}
        />
      ) : null}
    </>
  );
}
