"use client";

import React from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import "highlight.js/styles/github-dark.css";
import "katex/dist/katex.min.css";
import { escapeLiteralDollars } from "@/lib/markdown-math";
import { YouTubeEmbed } from "@/components/molecules/YouTubeEmbed/YouTubeEmbed";
import {
  getYouTubeVideoId,
  getYouTubeWatchUrl,
  isYouTubeUrl,
} from "@/components/molecules/YouTubeEmbed/helpers";

export function MarkdownContent({ value }: { value: string }) {
  const markdownContent = escapeLiteralDollars(value);

  return (
    <div className="markdown-output">
      <ReactMarkdown
        className="max-w-none"
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

// Module-level, so each render passes the same component types and React
// keeps the subtree (a playing video, a one-off "Load video") mounted.
const MARKDOWN_COMPONENTS: Components = {
  // Custom components for better rendering
  pre: ({ children, ...props }) => (
    <pre
      className="my-4 overflow-x-auto rounded-md bg-action-hover p-4"
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
          className="rounded bg-surface-hover px-1.5 py-0.5 font-mono text-body text-foreground-emphasis"
          {...props}
        >
          {children}
        </code>
      );
    }
    // Block code is handled by rehype-highlight
    return (
      <code className="font-mono text-body text-action-foreground" {...props}>
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
      className="my-4 border-l-4 border-info pl-4 text-foreground-body italic"
      {...props}
    >
      {children}
    </blockquote>
  ),
  table: ({ children, ...props }) => (
    <div className="my-4 overflow-x-auto">
      <table
        className="min-w-full divide-y divide-border-base border-y border-border-base"
        {...props}
      >
        {children}
      </table>
    </div>
  ),
  th: ({ children, ...props }) => (
    <th
      className="bg-surface-muted px-4 py-3.5 text-left text-caption font-semibold tracking-wider text-foreground-body uppercase"
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ children, ...props }) => (
    <td
      className="border-t border-border-base px-4 py-3.5 text-body text-foreground-secondary"
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
          className="mr-2 h-4 w-4 rounded border-border-strong text-info-foreground focus:ring-info disabled:cursor-not-allowed disabled:opacity-70"
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
      className={`text-foreground-body ${
        props.className?.includes("task-list-item") ? "flex items-start" : ""
      }`}
      {...props}
    >
      {children}
    </li>
  ),
  // Better heading styles
  h1: ({ children, ...props }) => (
    <h1 className="my-6 text-heading font-bold text-foreground-base" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }) => (
    <h2
      className="my-5 text-heading-md font-semibold text-foreground-emphasis"
      {...props}
    >
      {children}
    </h2>
  ),
  h3: ({ children, ...props }) => (
    <h3
      className="my-4 text-lead font-semibold text-foreground-emphasis"
      {...props}
    >
      {children}
    </h3>
  ),
  h4: ({ children, ...props }) => (
    <h4 className="my-3 text-lead font-medium text-foreground-body" {...props}>
      {children}
    </h4>
  ),
  h5: ({ children, ...props }) => (
    <h5
      className="my-2 text-body-lg font-medium text-foreground-body"
      {...props}
    >
      {children}
    </h5>
  ),
  h6: ({ children, ...props }) => (
    <h6
      className="my-2 text-body font-medium text-foreground-secondary"
      {...props}
    >
      {children}
    </h6>
  ),
  // Horizontal rule
  hr: ({ ...props }) => <hr className="my-6 border-border-strong" {...props} />,
  // Strikethrough (GFM)
  del: ({ children, ...props }) => (
    <del className="text-foreground-tertiary line-through" {...props}>
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
      // eslint-disable-next-line @next/next/no-img-element -- remote URL from any host, which next/image would need configured
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
      <p className="my-3 leading-relaxed text-foreground-body" {...props}>
        {children}
      </p>
    );
  },
};

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
            className="absolute top-0 left-0 rounded-lg shadow-md"
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
          className="absolute top-0 left-0 h-full w-full rounded-lg shadow-md"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
    </div>
  );
}
