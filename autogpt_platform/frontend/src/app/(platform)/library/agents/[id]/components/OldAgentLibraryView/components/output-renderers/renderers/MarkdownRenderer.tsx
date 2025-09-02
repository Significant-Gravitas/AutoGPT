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

export class MarkdownRenderer implements OutputRenderer {
  name = "MarkdownRenderer";
  priority = 35;

  private markdownPatterns = [
    /^#{1,6}\s+/m, // Headers
    /\*\*[^*]+\*\*/, // Bold
    /\*[^*]+\*/, // Italic
    /__[^_]+__/, // Bold alt
    /_[^_]+_/, // Italic alt
    /\[([^\]]+)\]\(([^)]+)\)/, // Links
    /```[\s\S]*?```/, // Code blocks
    /`[^`]+`/, // Inline code
    /^\s*[-*+]\s+/m, // Unordered lists
    /^\s*\d+\.\s+/m, // Ordered lists
    /^>\s+/m, // Blockquotes
    /\|.+\|/, // Tables
    /!\[([^\]]*)\]\(([^)]+)\)/, // Images
    /\$\$[\s\S]+?\$\$/, // Display math
    /\$[^$]+\$/, // Inline math
  ];

  canRender(value: any, metadata?: OutputMetadata): boolean {
    // Check metadata first
    if (
      metadata?.type === "markdown" ||
      metadata?.mimeType === "text/markdown" ||
      metadata?.mimeType === "text/x-markdown"
    ) {
      return true;
    }

    // Check if it's a string
    if (typeof value !== "string") {
      return false;
    }

    // Check filename extension
    if (metadata?.filename?.toLowerCase().endsWith(".md")) {
      return true;
    }

    // Count how many markdown patterns match
    let matchCount = 0;
    const requiredMatches = 2; // Need at least 2 patterns to consider it markdown

    for (const pattern of this.markdownPatterns) {
      if (pattern.test(value)) {
        matchCount++;
        if (matchCount >= requiredMatches) {
          return true;
        }
      }
    }

    return false;
  }

  render(value: any, _metadata?: OutputMetadata): React.ReactNode {
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
                className="text-blue-600 underline decoration-1 underline-offset-2 transition-colors hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
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
              if (src && this.isVideoUrl(src)) {
                return this.renderVideoEmbed(src);
              }

              return (
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
              if (
                typeof children === "string" &&
                this.isVideoUrl(children.trim())
              ) {
                return this.renderVideoEmbed(children.trim());
              }

              // Check for video URLs in link children
              if (React.Children.count(children) === 1) {
                const child = React.Children.toArray(children)[0];
                if (React.isValidElement(child) && child.type === "a") {
                  const href = child.props.href;
                  if (href && this.isVideoUrl(href)) {
                    return this.renderVideoEmbed(href);
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

  getCopyContent(value: any, _metadata?: OutputMetadata): CopyContent | null {
    const markdownText = String(value);
    return {
      mimeType: "text/markdown",
      data: markdownText,
      alternativeMimeTypes: ["text/plain"],
      fallbackText: markdownText,
    };
  }

  getDownloadContent(
    value: any,
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

  isConcatenable(_value: any, _metadata?: OutputMetadata): boolean {
    return true;
  }

  private isVideoUrl(url: string): boolean {
    // YouTube patterns
    if (url.includes("youtube.com/watch") || url.includes("youtu.be/")) {
      return true;
    }
    // Vimeo patterns
    if (url.includes("vimeo.com/")) {
      return true;
    }
    // Direct video file extensions
    const videoExtensions = [".mp4", ".webm", ".ogg", ".mov", ".avi"];
    return videoExtensions.some((ext) => url.toLowerCase().includes(ext));
  }

  private getVideoEmbedUrl(url: string): string | null {
    // YouTube
    const youtubeMatch = url.match(
      /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\s]+)/,
    );
    if (youtubeMatch) {
      return `https://www.youtube.com/embed/${youtubeMatch[1]}`;
    }

    // Vimeo
    const vimeoMatch = url.match(/vimeo\.com\/(\d+)/);
    if (vimeoMatch) {
      return `https://player.vimeo.com/video/${vimeoMatch[1]}`;
    }

    // Direct video URL
    const videoExtensions = [".mp4", ".webm", ".ogg", ".mov"];
    if (videoExtensions.some((ext) => url.toLowerCase().includes(ext))) {
      return url;
    }

    return null;
  }

  private renderVideoEmbed(url: string): React.ReactNode {
    const embedUrl = this.getVideoEmbedUrl(url);

    if (!embedUrl) {
      // Fallback to link if we can't embed
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

    // Check if it's a direct video file
    const videoExtensions = [".mp4", ".webm", ".ogg", ".mov"];
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

    // YouTube or Vimeo embed
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
}
