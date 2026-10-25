"use client";

import React from "react";
import ReactMarkdown, { type Components } from "react-markdown";
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
import { escapeCurrencyAmounts } from "@/lib/markdown-math";
import { YouTubeEmbed } from "@/components/molecules/YouTubeEmbed/YouTubeEmbed";
import {
  getYouTubeVideoId,
  getYouTubeWatchUrl,
  isYouTubeUrl,
} from "@/components/molecules/YouTubeEmbed/helpers";

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
  // A YouTube link is a video only when it names one; a channel or playlist
  // stays a link.
  if (isYouTubeUrl(url)) {
    return getYouTubeVideoId(url) !== null;
  }
  try {
    const parsed = new URL(url);
    if (
      parsed.hostname === "vimeo.com" ||
      parsed.hostname === "www.vimeo.com"
    ) {
      return true;
    }
  } catch {
    // If URL parsing fails, treat as not a Vimeo URL.
  }
  return videoExtensions.some((ext) => url.toLowerCase().includes(ext));
}

function getVideoEmbedUrl(url: string): string | null {
  const vimeoMatch = url.match(/vimeo\.com\/(\d+)/);
  if (vimeoMatch) {
    return `https://player.vimeo.com/video/${vimeoMatch[1]}`;
  }

  if (videoExtensions.some((ext) => url.toLowerCase().includes(ext))) {
    return url;
  }

  return null;
}

function renderPlainLink(url: string, label: React.ReactNode = url) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="text-info-foreground underline"
    >
      {label}
    </a>
  );
}

// Set inside a markdown link, so content rendered there doesn't add an
// anchor of its own (anchors can't nest).
const InsideLinkContext = React.createContext(false);

interface MarkdownVideoImageProps {
  src: string;
  alt?: string;
}

// An image whose source is a video, or a YouTube page that isn't one video
// (shown as a link, since an <img> would fetch it before consent). Inside a
// link only its text is shown: a player or a second link can't sit in an
// anchor.
function MarkdownVideoImage({ src, alt }: MarkdownVideoImageProps) {
  const insideLink = React.useContext(InsideLinkContext);
  if (insideLink) return <>{alt || src}</>;
  if (isVideoUrl(src)) return renderVideoEmbed(src);
  // Written without a scheme, the link would resolve against this site.
  return renderPlainLink(getYouTubeWatchUrl(src) ?? src, alt || src);
}

function renderVideoEmbed(url: string): React.ReactNode {
  if (getYouTubeVideoId(url)) {
    return (
      <div className="my-4">
        <div className="relative aspect-video">
          <YouTubeEmbed
            url={url}
            className="absolute left-0 top-0 rounded-lg shadow-md"
          />
        </div>
      </div>
    );
  }

  const embedUrl = getVideoEmbedUrl(url);

  if (!embedUrl) {
    return renderPlainLink(url);
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

// Module-level, so each render passes the same component types and React
// keeps the subtree (a playing video, a one-off "Load video") mounted.
const MARKDOWN_COMPONENTS: Components = {
  // Custom components for better rendering
  pre: ({ children, ...props }) => (
    <pre
      // TODO(tokens): no exact role for bg-gray-900 #111827
      className="my-4 overflow-x-auto rounded-md bg-gray-900 p-4"
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
          // TODO(tokens): no exact role for bg-gray-100 #f3f4f6, text-gray-800 #1f2937
          className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-sm text-gray-800"
          {...props}
        >
          {children}
        </code>
      );
    }
    // Block code is handled by rehype-highlight
    return (
      <code
        // TODO(tokens): no exact role for text-gray-100 #f3f4f6
        className="font-mono text-sm text-gray-100"
        {...props}
      >
        {children}
      </code>
    );
  },
  a: ({ children, href, ...props }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-foreground-base underline decoration-1 underline-offset-2 transition-colors"
      {...props}
    >
      <InsideLinkContext.Provider value={true}>
        {children}
      </InsideLinkContext.Provider>
    </a>
  ),
  blockquote: ({ children, ...props }) => (
    <blockquote
      // TODO(tokens): no exact role for text-gray-700 #374151
      className="my-4 border-l-4 border-info pl-4 italic text-gray-700"
      {...props}
    >
      {children}
    </blockquote>
  ),
  table: ({ children, ...props }) => (
    <div className="my-4 overflow-x-auto">
      <table
        // TODO(tokens): no exact role for divide-gray-200 #e5e7eb, border-gray-200 #e5e7eb; #e5e7eb is sidebar-border, the sidebar only
        className="min-w-full divide-y divide-gray-200 border-y border-gray-200"
        {...props}
      >
        {children}
      </table>
    </div>
  ),
  th: ({ children, ...props }) => (
    <th
      // TODO(tokens): no exact role for bg-gray-50 #f9fafb, text-gray-700 #374151
      className="bg-gray-50 px-4 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-gray-700"
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ children, ...props }) => (
    <td
      // TODO(tokens): no exact role for border-gray-200 #e5e7eb, text-gray-600 #4b5563; #e5e7eb is sidebar-border, the sidebar only
      className="border-t border-gray-200 px-4 py-3.5 text-sm text-gray-600"
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
          // TODO(tokens): no exact role for border-gray-300 #d1d5db
          className="mr-2 h-4 w-4 rounded border-gray-300 text-info-foreground focus:ring-info disabled:cursor-not-allowed disabled:opacity-70"
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
        props.className?.includes("contains-task-list") ? "list-none pl-0" : ""
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
      // TODO(tokens): no exact role for text-gray-700 #374151
      className={`text-gray-700 ${
        props.className?.includes("task-list-item") ? "flex items-start" : ""
      }`}
      {...props}
    >
      {children}
    </li>
  ),
  // Better heading styles
  h1: ({ children, ...props }) => (
    <h1
      // TODO(tokens): no exact role for text-gray-900 #111827
      className="my-6 text-3xl font-bold text-gray-900"
      {...props}
    >
      {children}
    </h1>
  ),
  h2: ({ children, ...props }) => (
    <h2
      // TODO(tokens): no exact role for text-gray-800 #1f2937
      className="my-5 text-2xl font-semibold text-gray-800"
      {...props}
    >
      {children}
    </h2>
  ),
  h3: ({ children, ...props }) => (
    <h3
      // TODO(tokens): no exact role for text-gray-800 #1f2937
      className="my-4 text-xl font-semibold text-gray-800"
      {...props}
    >
      {children}
    </h3>
  ),
  h4: ({ children, ...props }) => (
    <h4
      // TODO(tokens): no exact role for text-gray-700 #374151
      className="my-3 text-lg font-medium text-gray-700"
      {...props}
    >
      {children}
    </h4>
  ),
  h5: ({ children, ...props }) => (
    <h5
      // TODO(tokens): no exact role for text-gray-700 #374151
      className="my-2 text-base font-medium text-gray-700"
      {...props}
    >
      {children}
    </h5>
  ),
  h6: ({ children, ...props }) => (
    <h6
      // TODO(tokens): no exact role for text-gray-600 #4b5563
      className="my-2 text-sm font-medium text-gray-600"
      {...props}
    >
      {children}
    </h6>
  ),
  // Horizontal rule
  hr: ({ ...props }) => (
    <hr
      // TODO(tokens): no exact role for border-gray-300 #d1d5db
      className="my-6 border-gray-300"
      {...props}
    />
  ),
  // Strikethrough (GFM)
  del: ({ children, ...props }) => (
    <del
      // TODO(tokens): no exact role for text-gray-500 #6b7280
      className="text-gray-500 line-through"
      {...props}
    >
      {children}
    </del>
  ),
  // Image handling
  img: ({ src, alt, ...props }) => {
    if (
      src &&
      typeof src === "string" &&
      (isVideoUrl(src) || isYouTubeUrl(src))
    ) {
      return <MarkdownVideoImage src={src} alt={alt} />;
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
      if (
        React.isValidElement<{ href?: string }>(child) &&
        child.type === "a"
      ) {
        const href = child.props.href;
        if (href && isVideoUrl(href)) {
          return renderVideoEmbed(href);
        }
      }
    }

    return (
      <p
        // TODO(tokens): no exact role for text-gray-700 #374151
        className="my-3 leading-relaxed text-gray-700"
        {...props}
      >
        {children}
      </p>
    );
  },
};

function renderMarkdown(
  value: unknown,
  _metadata?: OutputMetadata,
): React.ReactNode {
  const markdownContent = escapeCurrencyAmounts(String(value));

  return (
    <div className="markdown-output">
      <ReactMarkdown
        className="prose prose-sm max-w-none"
        remarkPlugins={[
          remarkGfm, // GitHub Flavored Markdown (tables, task lists, strikethrough)
          [remarkMath, { singleDollarTextMath: true }], // Math support for LaTeX
        ]}
        rehypePlugins={[
          [rehypeKatex, { strict: false }], // Render math with KaTeX
          rehypeHighlight, // Syntax highlighting for code blocks
          rehypeSlug, // Add IDs to headings
          [rehypeAutolinkHeadings, { behavior: "wrap" }], // Make headings clickable
        ]}
        components={MARKDOWN_COMPONENTS}
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
