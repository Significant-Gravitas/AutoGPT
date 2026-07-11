"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  markdown: string;
  // Docs page the markdown came from — used to resolve relative *links*.
  // (Images are already rewritten to our same-origin proxy in cleanEntryMarkdown.)
  baseUrl?: string;
}

function resolveHref(href: string | undefined, base?: string) {
  if (!href || !base) return href;
  try {
    return new URL(href, base).href;
  } catch {
    return href;
  }
}

export function ChangelogMarkdownContent({ markdown, baseUrl }: Props) {
  return (
    <ReactMarkdown
      className="prose prose-sm max-w-none prose-headings:text-foreground prose-p:text-muted-foreground prose-a:text-accent prose-a:no-underline hover:prose-a:underline prose-strong:text-foreground prose-li:text-muted-foreground prose-img:rounded-lg prose-img:shadow-md"
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ node: _node, children, href, ...props }) => (
          <a
            href={resolveHref(href, baseUrl)}
            target="_blank"
            rel="noopener noreferrer"
            {...props}
          >
            {children}
          </a>
        ),
        img: ({ node: _node, src, alt, ...props }) => (
          <img
            src={typeof src === "string" ? src : undefined}
            alt={alt || ""}
            className="my-4 h-auto max-w-full rounded-lg shadow-md"
            loading="lazy"
            {...(props as React.ImgHTMLAttributes<HTMLImageElement>)}
          />
        ),
      }}
    >
      {markdown}
    </ReactMarkdown>
  );
}
