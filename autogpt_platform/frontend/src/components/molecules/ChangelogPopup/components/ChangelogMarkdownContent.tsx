"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  markdown: string;
  // Docs page the markdown was fetched from — used to resolve relative
  // links/images (e.g. `../assets/x.png`) against the docs origin.
  baseUrl?: string;
}

function resolveUrl(url: string | undefined, base?: string) {
  if (!url || !base) return url;
  try {
    return new URL(url, base).href;
  } catch {
    return url;
  }
}

export function ChangelogMarkdownContent({ markdown, baseUrl }: Props) {
  return (
    <ReactMarkdown
      className="prose prose-sm prose-headings:text-foreground prose-p:text-muted-foreground prose-a:text-accent prose-a:no-underline hover:prose-a:underline prose-strong:text-foreground prose-img:rounded-lg prose-img:shadow-md max-w-none"
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ node: _node, children, href, ...props }) => (
          <a
            href={resolveUrl(href, baseUrl)}
            target="_blank"
            rel="noopener noreferrer"
            {...props}
          >
            {children}
          </a>
        ),
        img: ({ node: _node, src, alt, ...props }) => (
          <img
            src={resolveUrl(typeof src === "string" ? src : undefined, baseUrl)}
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
