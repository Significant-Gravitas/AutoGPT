"use client";

import { MapAwarePre } from "@/components/contextual/MapEmbed/MapAwarePre";
import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import { escapeLiteralDollars } from "@/lib/markdown-math";
import { code } from "@/lib/streamdown-code-plugin";
import { mermaid } from "@/lib/streamdown-mermaid-plugin";
import { cn } from "@/lib/utils";
import { cjk } from "@streamdown/cjk";
import { createMathPlugin, type MathPlugin } from "@streamdown/math";
import type { UIMessage } from "ai";
import { defaultSchema } from "hast-util-sanitize";
import "katex/dist/katex.min.css";
import type { ComponentProps } from "react";
import rehypeKatex from "rehype-katex";
import {
  defaultRehypePlugins,
  defaultRemarkPlugins,
  parseMarkdownIntoBlocks,
  Streamdown,
} from "streamdown";
import { remarkAppLinks } from "./app-links";
import { MessageLink } from "./components/MessageLink/MessageLink";
import { rehypeLiteralTags } from "./literal-tags";
import {
  createComponentsMerger,
  type StreamdownComponents,
} from "./streamdown-components";

type Props = ComponentProps<typeof Streamdown> & {
  from: UIMessage["role"];
};

export function StreamdownResponse({
  className,
  children,
  components,
  from,
  ...props
}: Props) {
  // Merged on every render, as a plain call would be: the merger itself keeps
  // the result stable, and spots an entry the caller replaced in place, which
  // a result cached on `components` would miss.
  const mergedComponents = useReadEveryRender(() =>
    withDefaultComponents(components),
  );
  return (
    <Streamdown
      className={cn(
        "size-full [&_pre]:!bg-surface [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        "[&_a]:text-info [&_a]:no-underline [&_a]:hover:underline",
        // Raycast-style markdown tables: a rounded card with a quiet small-caps
        // header, hairline row separators and tabular figures.
        "[&_table]:w-full [&_table]:border-separate [&_table]:border-spacing-0 [&_table]:border-0 [&_table]:text-body-sm [&_table]:leading-inherit [&_table]:tabular-nums",
        "[&_tbody]:bg-transparent [&_thead]:bg-message-table-header",
        // Streamdown merges className again with its own tailwind-merge
        // (default config), which reads the size roles as colours and drops
        // them beside a colour in the same scope; the sizes therefore sit in
        // their own [&_table_*] scope. th and inline code keep the 20px
        // line-height their own text-sm gives them.
        "[&_table_td]:text-body-sm [&_table_td_code]:text-caption [&_table_td_code]:leading-5 [&_table_th]:text-caption-sm [&_table_th]:leading-5",
        "[&_th]:h-9 [&_th]:border-b [&_th]:border-border-base [&_th]:px-4 [&_th]:py-0 [&_th]:text-left [&_th]:font-medium [&_th]:tracking-eyebrow [&_th]:whitespace-nowrap [&_th]:uppercase",
        "[&_th]:text-shadow-message-table-indigo [&_th:first-child]:text-shadow-message-table-violet [&_th:last-child]:text-shadow-message-table-blue",
        "[&_td]:border-b [&_td]:border-separator [&_td]:px-4 [&_td]:py-2 [&_td]:align-top [&_td]:text-foreground-body [&_td]:transition-colors",
        "[&_td_strong]:font-medium [&_td_strong]:text-foreground-base",
        "[&_td_code]:rounded [&_td_code]:bg-surface-hover [&_td_code]:px-1 [&_td_code]:py-px [&_td_code]:whitespace-nowrap [&_td_code]:text-foreground-body",
        "[&_tbody_tr:last-child_td]:border-b-0",
        "[&_tbody_tr:hover_td]:bg-surface-muted",
        className,
      )}
      components={mergedComponents}
      plugins={STREAMDOWN_PLUGINS}
      remarkPlugins={REMARK_PLUGINS}
      rehypePlugins={
        from === "user" ? USER_REHYPE_PLUGINS : REPLY_REHYPE_PLUGINS
      }
      linkSafety={LINK_SAFETY}
      parseMarkdownIntoBlocksFn={parseEscapedBlocks}
      {...props}
    >
      {children}
    </Streamdown>
  );
}

// Single-dollar math is off by default in @streamdown/math, so "$x^2$" reached the
// reader raw; escapeLiteralDollars keeps prices and shell text out of its formulas.
// KaTeX renders text real LaTeX would reject, such as Bengali in a formula, but
// its default strict mode console.warns each time, and console capture filed
// every warning as a Sentry issue (BUILDER-94S). The plugin has no option for
// strict mode, so its rehype step is rebuilt with it off.
const math: MathPlugin = {
  ...createMathPlugin({ singleDollarTextMath: true }),
  rehypePlugin: [
    rehypeKatex,
    { errorColor: "var(--color-foreground-tertiary)", strict: "ignore" },
  ],
};

// Streamdown renders each block on its own, so each is escaped on its own, and a
// streamed reply re-parses only the block still growing.
const ESCAPED_BLOCKS = new Map<string, string>();
const ESCAPED_BLOCKS_LIMIT = 500;

function parseEscapedBlocks(markdown: string): string[] {
  return parseMarkdownIntoBlocks(markdown).map(escapeBlock);
}

function escapeBlock(block: string): string {
  const cached = ESCAPED_BLOCKS.get(block);
  if (cached !== undefined) {
    // Re-inserting keeps the blocks in use last in line for eviction.
    ESCAPED_BLOCKS.delete(block);
    ESCAPED_BLOCKS.set(block, cached);
    return cached;
  }
  const escaped = escapeLiteralDollars(block);
  if (ESCAPED_BLOCKS.size >= ESCAPED_BLOCKS_LIMIT) {
    ESCAPED_BLOCKS.delete(ESCAPED_BLOCKS.keys().next().value!);
  }
  ESCAPED_BLOCKS.set(block, escaped);
  return escaped;
}

// Streamdown re-parses every block of a message when the identity of a
// component in `components` changes, and React remounts any element whose
// component type changes, so everything handed to Streamdown is built once.
function MessageTable({ children, ...tableProps }: ComponentProps<"table">) {
  // Tables can be wider than the chat column (long URLs, many columns).
  // Wrap in an overflow-x-auto div so the page never grows past the
  // chat container — the table scrolls internally instead.
  return (
    <div className="my-4 max-w-full overflow-x-auto rounded-xl border border-border-base bg-surface shadow-message-table">
      <table {...tableProps}>{children}</table>
    </div>
  );
}

const STREAMDOWN_PLUGINS = { code, mermaid, math, cjk };

// Streamdown memoises every block on the identity of this list, so it is built
// once here.
const REMARK_PLUGINS = [...Object.values(defaultRemarkPlugins), remarkAppLinks];

// Server-only tags CommonMark reads as HTML (the rest carry an underscore, which
// already keeps them text) are dropped by the sanitiser; the text inside them shows.
const SERVER_ONLY_TAGS = ["routines", "thinking"];

// The user's bubble shows any tag they typed; Otto's keeps every element the
// sanitiser keeps. Streamdown's own raw, sanitise and harden steps run after.
const USER_REHYPE_PLUGINS: Props["rehypePlugins"] = [
  [rehypeLiteralTags, { keep: SERVER_ONLY_TAGS }],
  ...Object.values(defaultRehypePlugins),
];
const REPLY_REHYPE_PLUGINS: Props["rehypePlugins"] = [
  [
    rehypeLiteralTags,
    { keep: [...(defaultSchema.tagNames ?? []), ...SERVER_ONLY_TAGS] },
  ],
  ...Object.values(defaultRehypePlugins),
];

// MessageLink asks before leaving the site itself; Streamdown's own check
// would draw every link as a button without an href.
const LINK_SAFETY = { enabled: false };

const DEFAULT_COMPONENTS: StreamdownComponents = {
  table: MessageTable,
  pre: MapAwarePre,
  a: MessageLink,
};

// A caller may replace any default but the link, which keeps the dialog in
// front of every link to another site.
const mergeDefaultComponents = createComponentsMerger((components) => ({
  ...DEFAULT_COMPONENTS,
  ...components,
  a: MessageLink,
}));

function withDefaultComponents(components: StreamdownComponents | undefined) {
  return components ? mergeDefaultComponents(components) : DEFAULT_COMPONENTS;
}
