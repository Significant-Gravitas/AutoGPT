import { resolvePackagePath } from "@/components/contextual/SkillPackage/helpers";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { prepareSkillBody } from "./helpers";

interface Props {
  body: string;
  title: string;
  packagePaths: string[];
  onOpenFile: (path: string) => void;
}

export function SkillBody({ body, title, packagePaths, onOpenFile }: Props) {
  const withFileLinks: Components = {
    ...components,
    a: ({ children, href }) => (
      <BodyLink href={href} packagePaths={packagePaths} onOpenFile={onOpenFile}>
        {children}
      </BodyLink>
    ),
  };

  return (
    <div className="text-body-md leading-6 text-foreground-body [overflow-wrap:anywhere]">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={withFileLinks}>
        {prepareSkillBody(body, title)}
      </ReactMarkdown>
    </div>
  );
}

interface BodyLinkProps {
  href: string | undefined;
  packagePaths: string[];
  onOpenFile: (path: string) => void;
  children: React.ReactNode;
}

/** A link to a file the package ships opens the viewer; the marketplace never
 *  served those paths, so following one used to 404. */
function BodyLink({ href, packagePaths, onOpenFile, children }: BodyLinkProps) {
  const path = resolvePackagePath(href, packagePaths);
  if (path !== null) {
    return (
      <button
        type="button"
        onClick={() => onOpenFile(path)}
        className={`${LINK_CLASS} font-mono text-body-sm leading-inherit`}
      >
        {children}
      </button>
    );
  }

  const isExternal = /^https?:\/\//i.test(href ?? "");
  return (
    <a
      href={href}
      rel="noreferrer"
      {...(isExternal ? { target: "_blank" } : {})}
      className={LINK_CLASS}
    >
      {children}
    </a>
  );
}

const LINK_CLASS =
  // TODO(tokens): no exact role: hover:text-violet-600 #7C3AED
  "rounded-sm text-foreground-strong underline underline-offset-2 transition-colors hover:text-violet-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border-strong focus-visible:ring-offset-2";

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
    <strong className="font-semibold text-foreground-strong">{children}</strong>
  ),
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
    <th className="border-b border-border-base py-1.5 pr-4 text-left align-top font-medium text-foreground-strong">
      {children}
    </th>
  ),
  td: ({ children }) => <td className="py-1.5 pr-4 align-top">{children}</td>,
  code: ({ children }) => (
    <code className="break-words rounded bg-surface-hover px-1 py-0.5 font-mono text-xs">
      {children}
    </code>
  ),
  // The fenced block styles the frame; the code inside keeps only its font.
  pre: ({ children }) => (
    <pre className="mb-3 overflow-x-auto rounded-md bg-surface-muted p-3 text-body-sm leading-inherit [&_code]:bg-transparent [&_code]:p-0">
      {children}
    </pre>
  ),
};

const HEADING_CLASS: Record<number, string> = {
  3: "mb-2 mt-6 text-base font-semibold text-foreground-strong first:mt-0",
  4: "mb-2 mt-5 text-body-md leading-inherit font-semibold text-foreground-strong first:mt-0",
  5: "mb-1.5 mt-4 text-sm font-medium text-foreground-emphasis first:mt-0",
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
