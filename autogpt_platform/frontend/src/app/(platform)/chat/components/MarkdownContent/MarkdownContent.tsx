"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

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

export function MarkdownContent({ content, className }: MarkdownContentProps) {
  return (
    <div className={cn("markdown-content", className)}>
      <ReactMarkdown
        skipHtml={true}
        remarkPlugins={[remarkGfm]}
        components={{
          code: ({ children, className, ...props }: CodeProps) => {
            const isInline = !className?.includes("language-");
            if (isInline) {
              return (
                <code
                  className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-sm text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <code
                className="font-mono text-sm text-zinc-100 dark:text-zinc-200"
                {...props}
              >
                {children}
              </code>
            );
          },
          pre: ({ children, ...props }) => (
            <pre
              className="my-2 overflow-x-auto rounded-md bg-zinc-900 p-3 dark:bg-zinc-950"
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
              className="text-purple-600 underline decoration-1 underline-offset-2 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300"
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
                  className="mr-2 h-4 w-4 rounded border-zinc-300 text-purple-600 focus:ring-purple-500 disabled:cursor-not-allowed disabled:opacity-70 dark:border-zinc-600"
                  disabled
                  {...props}
                />
              );
            }
            return <input {...props} />;
          },
          blockquote: ({ children, ...props }) => (
            <blockquote
              className="my-2 border-l-4 border-zinc-300 pl-3 italic text-zinc-700 dark:border-zinc-600 dark:text-zinc-300"
              {...props}
            >
              {children}
            </blockquote>
          ),
          h1: ({ children, ...props }) => (
            <h1
              className="my-2 text-xl font-bold text-zinc-900 dark:text-zinc-100"
              {...props}
            >
              {children}
            </h1>
          ),
          h2: ({ children, ...props }) => (
            <h2
              className="my-2 text-lg font-semibold text-zinc-800 dark:text-zinc-200"
              {...props}
            >
              {children}
            </h2>
          ),
          h3: ({ children, ...props }) => (
            <h3
              className="my-1 text-base font-semibold text-zinc-800 dark:text-zinc-200"
              {...props}
            >
              {children}
            </h3>
          ),
          h4: ({ children, ...props }) => (
            <h4
              className="my-1 text-sm font-medium text-zinc-700 dark:text-zinc-300"
              {...props}
            >
              {children}
            </h4>
          ),
          h5: ({ children, ...props }) => (
            <h5
              className="my-1 text-sm font-medium text-zinc-700 dark:text-zinc-300"
              {...props}
            >
              {children}
            </h5>
          ),
          h6: ({ children, ...props }) => (
            <h6
              className="my-1 text-xs font-medium text-zinc-600 dark:text-zinc-400"
              {...props}
            >
              {children}
            </h6>
          ),
          p: ({ children, ...props }) => (
            <p className="my-2 leading-relaxed" {...props}>
              {children}
            </p>
          ),
          hr: ({ ...props }) => (
            <hr
              className="my-3 border-zinc-300 dark:border-zinc-700"
              {...props}
            />
          ),
          table: ({ children, ...props }) => (
            <div className="my-2 overflow-x-auto">
              <table
                className="min-w-full divide-y divide-zinc-200 rounded border border-zinc-200 dark:divide-zinc-700 dark:border-zinc-700"
                {...props}
              >
                {children}
              </table>
            </div>
          ),
          th: ({ children, ...props }) => (
            <th
              className="bg-zinc-50 px-3 py-2 text-left text-xs font-semibold text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
              {...props}
            >
              {children}
            </th>
          ),
          td: ({ children, ...props }) => (
            <td
              className="border-t border-zinc-200 px-3 py-2 text-sm dark:border-zinc-700"
              {...props}
            >
              {children}
            </td>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
