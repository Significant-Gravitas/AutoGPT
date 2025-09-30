"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";
import "highlight.js/styles/github-dark.css";
import "katex/dist/katex.min.css";

const markdownPatterns = [
  /```[\s\S]*?```/u, // Fenced code blocks (check first)
  /^#{1,6}\s+\S+/gmu, // ATX headers (require content)
  /\*\*[^*\n]+?\*\*/u, // **bold**
  /__(?!_)[^_\n]+?__(?!_)/u, // __bold__ (avoid ___/snake_case_)
  /(?<!\*)\*(?!\*)(?:[^*\n]|(?<=\\)\*)+?(?<!\\)\*(?!\*)/u, // *italic* (try to avoid **)
  /(?<!_)_(?!_)(?:[^_\n]|(?<=\\)_)+?(?<!\\)_(?!_)/u, // _italic_ with guards
  /\[([^\]\n]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/u, // Links with optional title (simple)
  /!\[([^\]\n]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/u, // Images with optional title (simple)
  /`[^`\n]+`/u, // Inline code
  /^(?:\s*[-*+]\s+\S.*)(?:\n\s*[-*+]\s+\S.*)+$/gmu, // UL list (≥2 items)
  /^(?:\s*\d+\.\s+\S.*)(?:\n\s*\d+\.\s+\S.*)+$/gmu, // OL list (≥2 items)
  /^>\s+\S.*/gm, // Blockquotes
  /^\|[^|\n]+(\|[^|\n]+)+\|\s*$/gm, // Table row (at least two cells)
  /^\s*\|(?:\s*:?[-=]{3,}\s*\|)+\s*$/gm, // Table separator row
  /\$\$[\s\S]+?\$\$/u, // Display math
  /(?<!\\)(?<!\w)\$[^$\n]+?\$(?!\w)/u, // Inline math: avoid prices/ids
];

const videoExtensions = [".mp4", ".webm", ".ogg", ".mov", ".avi"];

function isVideoUrl(url: string): boolean {
  if (url.includes("youtube.com/watch") || url.includes("youtu.be/")) {
    return true;
  }
  if (url.includes("vimeo.com/")) {
    return true;
  }
  return videoExtensions.some((ext) => url.toLowerCase().includes(ext));
}

function getVideoEmbedUrl(url: string): string | null {
  const youtubeMatch = url.match(
    /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\s]+)/,
  );
  if (youtubeMatch) {
    return `https://www.youtube.com/embed/${youtubeMatch[1]}`;
  }

  const vimeoMatch = url.match(/vimeo\.com\/(\d+)/);
  if (vimeoMatch) {
    return `https://player.vimeo.com/video/${vimeoMatch[1]}`;
  }

  if (videoExtensions.some((ext) => url.toLowerCase().includes(ext))) {
    return url;
  }

  return null;
}

function renderVideoEmbed(url: string): React.ReactNode {
  const embedUrl = getVideoEmbedUrl(url);

  if (!embedUrl) {
    return (
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 underline"
      >
        {url}
      </a>
    );
  }

  if (videoExtensions.some((ext) => embedUrl.toLowerCase().includes(ext))) {
    return (
      <div className="my-4">
        <video
          controls
          className="w-full max-w-2xl rounded-lg shadow-md"
          preload="metadata"
        >
          <source src={embedUrl} />
          Your browser does not support the video tag.
        </video>
      </div>
    );
  }

  return (
    <div className="my-4">
      <div className="relative aspect-video">
        <iframe
          src={embedUrl}
          title="Embedded video player"
          className="absolute left-0 top-0 h-full w-full rounded-lg shadow-md"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
    </div>
  );
}

function canRenderMarkdown(value: unknown, metadata?: OutputMetadata): boolean {
  if (
    metadata?.type === "markdown" ||
    metadata?.mimeType === "text/markdown" ||
    metadata?.mimeType === "text/x-markdown"
  ) {
    return true;
  }

  if (typeof value !== "string") {
    return false;
  }

  if (metadata?.filename?.toLowerCase().endsWith(".md")) {
    return true;
  }

  let matchCount = 0;
  const requiredMatches = 2;

  for (const pattern of markdownPatterns) {
    if (pattern.test(value)) {
      matchCount++;
      if (matchCount >= requiredMatches) {
        return true;
      }
    }
  }

  return false;
}

function renderMarkdown(
  value: unknown,
  _metadata?: OutputMetadata,
): React.ReactNode {
  const markdownContent = String(value);

  return (
    <div className="markdown-output">
      <ReactMarkdown
        className="prose prose-sm dark:prose-invert max-w-none"
        remarkPlugins={[
          remarkGfm, // GitHub Flavored Markdown (tables, task lists, strikethrough)
          remarkMath, // Math support for LaTeX
        ]}
        rehypePlugins={[
          rehypeKatex, // Render math with KaTeX
          rehypeHighlight, // Syntax highlighting for code blocks
          rehypeSlug, // Add IDs to headings
          [rehypeAutolinkHeadings, { behavior: "wrap" }], // Make headings clickable
        ]}
        components={{
          // Custom components for better rendering
          pre: ({ children, ...props }) => (
            <pre
              className="my-4 overflow-x-auto rounded-md bg-gray-900 p-4 dark:bg-gray-950"
              {...props}
            >
              {children}
            </pre>
          ),
          code: ({ children, ...props }: any) => {
            // Check if it's inline code by looking at the parent
            const isInline = !props.className?.includes("language-");
            if (isInline) {
              return (
                <code
                  className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-sm text-gray-800 dark:bg-gray-800 dark:text-gray-200"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            // Block code is handled by rehype-highlight
            return (
              <code className="font-mono text-sm text-gray-100" {...props}>
                {children}
              </code>
            );
          },
          a: ({ children, href, ...props }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-black underline decoration-1 underline-offset-2 transition-colors"
              {...props}
            >
              {children}
            </a>
          ),
          blockquote: ({ children, ...props }) => (
            <blockquote
              className="my-4 border-l-4 border-blue-500 pl-4 italic text-gray-700 dark:border-blue-400 dark:text-gray-300"
              {...props}
            >
              {children}
            </blockquote>
          ),
          table: ({ children, ...props }) => (
            <div className="my-4 overflow-x-auto">
              <table
                className="min-w-full divide-y divide-gray-200 rounded-lg border border-gray-200 dark:divide-gray-700 dark:border-gray-700"
                {...props}
              >
                {children}
              </table>
            </div>
          ),
          th: ({ children, ...props }) => (
            <th
              className="bg-gray-50 px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-700 dark:bg-gray-800 dark:text-gray-300"
              {...props}
            >
              {children}
            </th>
          ),
          td: ({ children, ...props }) => (
            <td
              className="border-t border-gray-200 px-4 py-3 text-sm text-gray-600 dark:border-gray-700 dark:text-gray-400"
              {...props}
            >
              {children}
            </td>
          ),
          // GitHub Flavored Markdown task lists
          input: ({ ...props }: any) => {
            if (props.type === "checkbox") {
              return (
                <input
                  type="checkbox"
                  className="mr-2 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-70"
                  disabled
                  {...props}
                />
              );
            }
            return <input {...props} />;
          },
          // Better list styling
          ul: ({ children, ...props }: any) => (
            <ul
              className={`my-4 list-disc space-y-2 pl-6 ${
                props.className?.includes("contains-task-list")
                  ? "list-none pl-0"
                  : ""
              }`}
              {...props}
            >
              {children}
            </ul>
          ),
          ol: ({ children, ...props }) => (
            <ol className="my-4 list-decimal space-y-2 pl-6" {...props}>
              {children}
            </ol>
          ),
          li: ({ children, ...props }: any) => (
            <li
              className={`text-gray-700 dark:text-gray-300 ${
                props.className?.includes("task-list-item")
                  ? "flex items-start"
                  : ""
              }`}
              {...props}
            >
              {children}
            </li>
          ),
          // Better heading styles
          h1: ({ children, ...props }) => (
            <h1
              className="my-6 text-3xl font-bold text-gray-900 dark:text-gray-100"
              {...props}
            >
              {children}
            </h1>
          ),
          h2: ({ children, ...props }) => (
            <h2
              className="my-5 text-2xl font-semibold text-gray-800 dark:text-gray-200"
              {...props}
            >
              {children}
            </h2>
          ),
          h3: ({ children, ...props }) => (
            <h3
              className="my-4 text-xl font-semibold text-gray-800 dark:text-gray-200"
              {...props}
            >
              {children}
            </h3>
          ),
          h4: ({ children, ...props }) => (
            <h4
              className="my-3 text-lg font-medium text-gray-700 dark:text-gray-300"
              {...props}
            >
              {children}
            </h4>
          ),
          h5: ({ children, ...props }) => (
            <h5
              className="my-2 text-base font-medium text-gray-700 dark:text-gray-300"
              {...props}
            >
              {children}
            </h5>
          ),
          h6: ({ children, ...props }) => (
            <h6
              className="my-2 text-sm font-medium text-gray-600 dark:text-gray-400"
              {...props}
            >
              {children}
            </h6>
          ),
          // Horizontal rule
          hr: ({ ...props }) => (
            <hr
              className="my-6 border-gray-300 dark:border-gray-700"
              {...props}
            />
          ),
          // Strikethrough (GFM)
          del: ({ children, ...props }) => (
            <del
              className="text-gray-500 line-through dark:text-gray-500"
              {...props}
            >
              {children}
            </del>
          ),
          // Image handling
          img: ({ src, alt, ...props }) => {
            // Check if it's a video URL pattern
            if (src && isVideoUrl(src)) {
              return renderVideoEmbed(src);
            }

            return (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={src}
                alt={alt}
                className="my-4 h-auto max-w-full rounded-lg shadow-md"
                loading="lazy"
                {...props}
              />
            );
          },
          // Custom paragraph to handle standalone video URLs
          p: ({ children, ...props }) => {
            // Check if paragraph contains just a video URL
            if (typeof children === "string" && isVideoUrl(children.trim())) {
              return renderVideoEmbed(children.trim());
            }

            // Check for video URLs in link children
            if (React.Children.count(children) === 1) {
              const child = React.Children.toArray(children)[0];
              if (React.isValidElement(child) && child.type === "a") {
                const href = child.props.href;
                if (href && isVideoUrl(href)) {
                  return renderVideoEmbed(href);
                }
              }
            }

            return (
              <p
                className="my-3 leading-relaxed text-gray-700 dark:text-gray-300"
                {...props}
              >
                {children}
              </p>
            );
          },
        }}
      >
        {markdownContent}
      </ReactMarkdown>
    </div>
  );
}

function getCopyContentMarkdown(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  const markdownText = String(value);
  return {
    mimeType: "text/markdown",
    data: markdownText,
    fallbackText: markdownText,
    alternativeMimeTypes: ["text/plain"],
  };
}

function getDownloadContentMarkdown(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const markdownText = String(value);
  const blob = new Blob([markdownText], { type: "text/markdown" });

  return {
    data: blob,
    filename: metadata?.filename || "output.md",
    mimeType: "text/markdown",
  };
}

function isConcatenableMarkdown(
  _value: unknown,
  _metadata?: OutputMetadata,
): boolean {
  return true;
}

export const markdownRenderer: OutputRenderer = {
  name: "MarkdownRenderer",
  priority: 35,
  canRender: canRenderMarkdown,
  render: renderMarkdown,
  getCopyContent: getCopyContentMarkdown,
  getDownloadContent: getDownloadContentMarkdown,
  isConcatenable: isConcatenableMarkdown,
};
