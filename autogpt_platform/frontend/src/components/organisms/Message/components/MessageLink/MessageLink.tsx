import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";
import { ExternalLinkButton } from "./components/ExternalLinkButton/ExternalLinkButton";
import { INCOMPLETE_LINK_HREF, isSameOriginLink } from "./helpers";

interface Props extends ComponentProps<"a"> {
  node?: unknown;
}

// Links to our own pages are real anchors, so middle-click, Ctrl/Cmd-click and
// the browser's "Open in new tab" work on them. Links to other sites stay
// buttons that always ask first (SECRT-2096).
export function MessageLink({
  children,
  className,
  href,
  node: _node,
  ...props
}: Props) {
  const isIncomplete = href === INCOMPLETE_LINK_HREF;
  const url = isIncomplete ? undefined : href;

  if (url && !isSameOriginLink(url)) {
    return (
      <ExternalLinkButton url={url} className={className}>
        {children}
      </ExternalLinkButton>
    );
  }

  return (
    <a
      {...props}
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      data-incomplete={isIncomplete}
      data-streamdown="link"
      // Doubled to outrank the chat's `[&_a]` colour, as WorkspaceLink does.
      className={cn("font-medium [&&]:text-primary [&&]:underline", className)}
    >
      {children}
    </a>
  );
}
