"use client";

import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { cn } from "@/lib/utils";
import { EyeSlash } from "@phosphor-icons/react";
import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

interface CodeProps extends React.HTMLAttributes<HTMLElement> {
  children?: React.ReactNode;
  className?: string;
}

interface ListProps extends React.HTMLAttributes<HTMLUListElement> {
  children?: React.ReactNode;
  className?: string;
}

interface ListItemProps extends React.HTMLAttributes<HTMLLIElement> {
  children?: React.ReactNode;
  className?: string;
}

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  type?: string;
}

/**
 * Converts a workspace:// URL to a proxy URL that routes through Next.js to the backend.
 * workspace://abc123 -> /api/proxy/api/workspace/files/abc123/download
 *
 * Uses the generated API URL helper and routes through the Next.js proxy
 * which handles authentication and proper backend routing.
 */
/**
 * URL transformer for ReactMarkdown.
 * Converts workspace:// URLs to proxy URLs that route through Next.js to the backend.
 * workspace://abc123 -> /api/proxy/api/workspace/files/abc123/download
 *
 * This is needed because ReactMarkdown sanitizes URLs and only allows
 * http, https, mailto, and tel protocols by default.
 */
function resolveWorkspaceUrl(src: string): string {
  if (src.startsWith("workspace://")) {
    const fileId = src.replace("workspace://", "");
    // Use the generated API URL helper to get the correct path
    const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
    // Route through the Next.js proxy (same pattern as customMutator for client-side)
    return `/api/proxy${apiPath}`;
  }
  return src;
}

/**
 * Check if the image URL is a workspace file (AI cannot see these yet).
 * After URL transformation, workspace files have URLs like /api/proxy/api/workspace/files/...
 */
function isWorkspaceImage(src: string | undefined): boolean {
  return src?.includes("/workspace/files/") ?? false;
}

/**
 * Custom image component that shows an indicator when the AI cannot see the image.
 * Note: src is already transformed by urlTransform, so workspace:// is now /api/workspace/...
 */
function MarkdownImage(props: Record<string, unknown>) {
  const src = props.src as string | undefined;
  const alt = props.alt as string | undefined;

  const aiCannotSee = isWorkspaceImage(src);

  // If no src, show a placeholder
  if (!src) {
    return (
      <span className="my-2 inline-block rounded border border-amber-200 bg-amber-50 px-2 py-1 text-sm text-amber-700">
        [Image: {alt || "missing src"}]
      </span>
    );
  }

  return (
    <span className="relative my-2 inline-block">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt || "Image"}
        className="h-auto max-w-full rounded-md border border-zinc-200"
        loading="lazy"
      />
      {aiCannotSee && (
        <span
          className="absolute bottom-2 right-2 flex items-center gap-1 rounded bg-black/70 px-2 py-1 text-xs text-white"
          title="The AI cannot see this image"
        >
          <EyeSlash size={14} />
          <span>AI cannot see this image</span>
        </span>
      )}
    </span>
  );
}

export function MarkdownContent({ content, className }: MarkdownContentProps) {
  return (
    <div className={cn("markdown-content", className)}>
      <ReactMarkdown
        skipHtml={true}
        remarkPlugins={[remarkGfm]}
        urlTransform={resolveWorkspaceUrl}
        components={{
          code: ({ children, className, ...props }: CodeProps) => {
            const isInline = !className?.includes("language-");
            if (isInline) {
              return (
                <code
                  className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-sm text-zinc-800"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <code className="font-mono text-sm text-zinc-100" {...props}>
                {children}
              </code>
            );
          },
          pre: ({ children, ...props }) => (
            <pre
              className="my-2 overflow-x-auto rounded-md bg-zinc-900 p-3"
              {...props}
            >
              {children}
            </pre>
          ),
          a: ({ children, href, ...props }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-purple-600 underline decoration-1 underline-offset-2 hover:text-purple-700"
              {...props}
            >
              {children}
            </a>
          ),
          strong: ({ children, ...props }) => (
            <strong className="font-semibold" {...props}>
              {children}
            </strong>
          ),
          em: ({ children, ...props }) => (
            <em className="italic" {...props}>
              {children}
            </em>
          ),
          del: ({ children, ...props }) => (
            <del className="line-through opacity-70" {...props}>
              {children}
            </del>
          ),
          ul: ({ children, ...props }: ListProps) => (
            <ul
              className={cn(
                "my-2 space-y-1 pl-6",
                props.className?.includes("contains-task-list")
                  ? "list-none pl-0"
                  : "list-disc",
              )}
              {...props}
            >
              {children}
            </ul>
          ),
          ol: ({ children, ...props }) => (
            <ol className="my-2 list-decimal space-y-1 pl-6" {...props}>
              {children}
            </ol>
          ),
          li: ({ children, ...props }: ListItemProps) => (
            <li
              className={cn(
                props.className?.includes("task-list-item")
                  ? "flex items-start"
                  : "",
              )}
              {...props}
            >
              {children}
            </li>
          ),
          input: ({ ...props }: InputProps) => {
            if (props.type === "checkbox") {
              return (
                <input
                  type="checkbox"
                  className="mr-2 h-4 w-4 rounded border-zinc-300 text-purple-600 focus:ring-purple-500 disabled:cursor-not-allowed disabled:opacity-70"
                  disabled
                  {...props}
                />
              );
            }
            return <input {...props} />;
          },
          blockquote: ({ children, ...props }) => (
            <blockquote
              className="my-2 border-l-4 border-zinc-300 pl-3 italic text-zinc-700"
              {...props}
            >
              {children}
            </blockquote>
          ),
          h1: ({ children, ...props }) => (
            <h1 className="my-2 text-xl font-bold text-zinc-900" {...props}>
              {children}
            </h1>
          ),
          h2: ({ children, ...props }) => (
            <h2 className="my-2 text-lg font-semibold text-zinc-800" {...props}>
              {children}
            </h2>
          ),
          h3: ({ children, ...props }) => (
            <h3
              className="my-1 text-base font-semibold text-zinc-800"
              {...props}
            >
              {children}
            </h3>
          ),
          h4: ({ children, ...props }) => (
            <h4 className="my-1 text-sm font-medium text-zinc-700" {...props}>
              {children}
            </h4>
          ),
          h5: ({ children, ...props }) => (
            <h5 className="my-1 text-sm font-medium text-zinc-700" {...props}>
              {children}
            </h5>
          ),
          h6: ({ children, ...props }) => (
            <h6 className="my-1 text-xs font-medium text-zinc-600" {...props}>
              {children}
            </h6>
          ),
          p: ({ children, ...props }) => (
            <p className="my-2 leading-relaxed" {...props}>
              {children}
            </p>
          ),
          hr: ({ ...props }) => (
            <hr className="my-3 border-zinc-300" {...props} />
          ),
          table: ({ children, ...props }) => (
            <div className="my-2 overflow-x-auto">
              <table
                className="min-w-full divide-y divide-zinc-200 rounded border border-zinc-200"
                {...props}
              >
                {children}
              </table>
            </div>
          ),
          th: ({ children, ...props }) => (
            <th
              className="bg-zinc-50 px-3 py-2 text-left text-xs font-semibold text-zinc-700"
              {...props}
            >
              {children}
            </th>
          ),
          td: ({ children, ...props }) => (
            <td
              className="border-t border-zinc-200 px-3 py-2 text-sm"
              {...props}
            >
              {children}
            </td>
          ),
          img: ({ src, alt, ...props }) => (
            <MarkdownImage src={src} alt={alt} {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
