import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { prepareSkillBody } from "./helpers";

interface Props {
  body: string;
  title: string;
}

export function SkillBody({ body, title }: Props) {
  return (
    <div className="text-[15px] leading-6 text-zinc-700 [overflow-wrap:anywhere]">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {prepareSkillBody(body, title)}
      </ReactMarkdown>
    </div>
  );
}

const components: Components = {
  h1: ({ children }) => <Heading level={3}>{children}</Heading>,
  h2: ({ children }) => <Heading level={3}>{children}</Heading>,
  h3: ({ children }) => <Heading level={3}>{children}</Heading>,
  h4: ({ children }) => <Heading level={4}>{children}</Heading>,
  h5: ({ children }) => <Heading level={5}>{children}</Heading>,
  h6: ({ children }) => <Heading level={5}>{children}</Heading>,
  p: ({ children }) => <p className="mb-3">{children}</p>,
  ul: ({ children }) => (
    <ul className="mb-3 ml-5 list-disc space-y-1">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-3 ml-5 list-decimal space-y-1">{children}</ol>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-zinc-900">{children}</strong>
  ),
  a: ({ children, href }) => {
    const isExternal = /^https?:\/\//i.test(href ?? "");
    return (
      <a
        href={href}
        rel="noreferrer"
        {...(isExternal ? { target: "_blank" } : {})}
        className="text-zinc-900 underline underline-offset-2 transition-colors hover:text-violet-600"
      >
        {children}
      </a>
    );
  },
  // Creator markdown carries no dimensions, so the shift is contained by the
  // radius and max-width rather than eliminated.
  img: ({ src, alt }) => (
    <img
      src={typeof src === "string" ? src : undefined}
      alt={alt ?? ""}
      loading="lazy"
      className="my-3 h-auto max-w-full rounded-lg"
    />
  ),
  table: ({ children }) => (
    <div className="mb-3 overflow-x-auto">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border-b border-zinc-200 py-1.5 pr-4 text-left align-top font-medium text-zinc-900">
      {children}
    </th>
  ),
  td: ({ children }) => <td className="py-1.5 pr-4 align-top">{children}</td>,
  code: ({ children }) => (
    <code className="break-words rounded bg-zinc-100 px-1 py-0.5 font-mono text-xs">
      {children}
    </code>
  ),
  // The fenced block styles the frame; the code inside keeps only its font.
  pre: ({ children }) => (
    <pre className="mb-3 overflow-x-auto rounded-md bg-zinc-50 p-3 text-[13px] [&_code]:bg-transparent [&_code]:p-0">
      {children}
    </pre>
  ),
};

const HEADING_CLASS: Record<number, string> = {
  3: "mb-2 mt-6 text-base font-semibold text-zinc-900 first:mt-0",
  4: "mb-2 mt-5 text-[15px] font-semibold text-zinc-900 first:mt-0",
  5: "mb-1.5 mt-4 text-sm font-medium text-zinc-800 first:mt-0",
};

function Heading({
  level,
  children,
}: {
  level: 3 | 4 | 5;
  children: React.ReactNode;
}) {
  const Tag = `h${level}` as "h3" | "h4" | "h5";
  return <Tag className={HEADING_CLASS[level]}>{children}</Tag>;
}
