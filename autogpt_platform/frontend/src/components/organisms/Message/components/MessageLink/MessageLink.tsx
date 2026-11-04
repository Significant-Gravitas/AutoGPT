import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";
import { ExternalLinkButton } from "./components/ExternalLinkButton/ExternalLinkButton";
import { INCOMPLETE_LINK_HREF } from "./helpers";
import { useMessageLink } from "./useMessageLink";

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
  const { kind } = useMessageLink(href);

  if (kind === "external" && href) {
    return (
      <ExternalLinkButton url={href} title={props.title} className={className}>
        {children}
      </ExternalLinkButton>
    );
  }

  return (
    <a
      {...props}
      href={kind === "site" ? href : undefined}
      target="_blank"
      rel="noopener noreferrer"
      data-incomplete={href === INCOMPLETE_LINK_HREF}
      data-streamdown="link"
      // Doubled to outrank the chat's `[&_a]` colour, as WorkspaceLink does.
      className={cn("font-medium [&&]:text-action [&&]:underline", className)}
    >
      {children}
    </a>
  );
}
