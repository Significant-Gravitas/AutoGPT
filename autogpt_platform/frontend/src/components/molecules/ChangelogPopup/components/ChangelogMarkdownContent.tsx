"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function ChangelogMarkdownContent({ markdown }: { markdown: string }) {
  return (
    <ReactMarkdown
      className="prose prose-sm max-w-none prose-headings:text-foreground prose-p:text-muted-foreground prose-a:text-accent prose-a:no-underline hover:prose-a:underline prose-strong:text-foreground prose-img:rounded-lg prose-img:shadow-md"
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ children, href, ...props }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            {...props}
          >
            {children}
          </a>
        ),
        img: ({ src, alt, ...props }) => (
          <img
            src={src}
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
