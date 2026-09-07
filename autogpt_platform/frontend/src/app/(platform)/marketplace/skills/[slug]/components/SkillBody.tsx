import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  body: string;
}

export function SkillBody({ body }: Props) {
  return (
    <div className="text-sm leading-relaxed text-zinc-700">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {body}
      </ReactMarkdown>
    </div>
  );
}

// A SKILL.md is prose the reader is judging before they install it, so the
// heading levels inside it are demoted — the page already owns h1 and h2.
const components: Components = {
  h1: ({ children }) => (
    <h3 className="mb-2 mt-6 text-base font-semibold text-zinc-900 first:mt-0">
      {children}
    </h3>
  ),
  h2: ({ children }) => (
    <h4 className="mb-2 mt-6 text-sm font-semibold text-zinc-900 first:mt-0">
      {children}
    </h4>
  ),
  h3: ({ children }) => (
    <h5 className="mb-1.5 mt-4 text-sm font-medium text-zinc-800">
      {children}
    </h5>
  ),
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
  code: ({ children }) => (
    <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-xs">
      {children}
    </code>
  ),
};
