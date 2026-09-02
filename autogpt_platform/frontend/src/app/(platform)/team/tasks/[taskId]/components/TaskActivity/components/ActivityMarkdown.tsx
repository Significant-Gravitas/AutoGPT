import Link from "next/link";
import ReactMarkdown, { type Components } from "react-markdown";

const HEADING_CLASS = "mb-1 text-[13px] font-semibold text-zinc-900";

const MARKDOWN_COMPONENTS: Components = {
  h1: ({ children }) => <p className={HEADING_CLASS}>{children}</p>,
  h2: ({ children }) => <p className={HEADING_CLASS}>{children}</p>,
  h3: ({ children }) => <p className={HEADING_CLASS}>{children}</p>,
  p: ({ children }) => (
    <p className="mb-1.5 text-[13px] leading-5 text-zinc-700 last:mb-0">
      {children}
    </p>
  ),
  ul: ({ children }) => (
    <ul className="mb-1.5 ml-4 list-disc text-[13px] leading-5 text-zinc-700 last:mb-0">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-1.5 ml-4 list-decimal text-[13px] leading-5 text-zinc-700 last:mb-0">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="mb-0.5">{children}</li>,
  strong: ({ children }) => (
    <strong className="font-semibold text-zinc-900">{children}</strong>
  ),
  a: ({ href, children }) => {
    const url = href ?? "";
    const linkClass = "font-medium text-violet-600 hover:underline";
    if (url.startsWith("/")) {
      return (
        <Link href={url} className={linkClass}>
          {children}
        </Link>
      );
    }
    return (
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className={linkClass}
      >
        {children}
      </a>
    );
  },
  code: ({ children }) => (
    <code className="rounded bg-zinc-100 px-1 font-mono text-[12px] text-zinc-800">
      {children}
    </code>
  ),
  pre: ({ children }) => (
    <pre className="mb-1.5 overflow-x-auto rounded-lg bg-zinc-100 p-2 font-mono text-[12px] text-zinc-800 last:mb-0">
      {children}
    </pre>
  ),
};

/** Agent-written activity text is markdown; render it small and quiet so a
 *  quote reads as part of the feed, not a document dropped into it. */
export function ActivityMarkdown({ text }: { text: string }) {
  return <ReactMarkdown components={MARKDOWN_COMPONENTS}>{text}</ReactMarkdown>;
}
