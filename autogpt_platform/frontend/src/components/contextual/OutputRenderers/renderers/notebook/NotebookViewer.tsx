"use client";

import { CaretDown, CaretRight } from "@phosphor-icons/react";
import DOMPurify from "dompurify";
import Image from "next/image";
import { useEffect, useState } from "react";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { joinSource } from "./helpers";
import type { Notebook, NotebookCell, NotebookOutput } from "./types";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark.css";

function sanitizeNotebookMarkup(markup: string): string {
  return DOMPurify.sanitize(markup, {
    USE_PROFILES: { html: true, svg: true, svgFilters: true },
  });
}

function SanitizedNotebookMarkup({
  className,
  markup,
}: {
  className: string;
  markup: string;
}) {
  const [sanitizedMarkup, setSanitizedMarkup] = useState<string | null>(null);

  useEffect(() => {
    setSanitizedMarkup(sanitizeNotebookMarkup(markup));
  }, [markup]);

  if (sanitizedMarkup === null) return null;

  return (
    <div
      className={className}
      dangerouslySetInnerHTML={{ __html: sanitizedMarkup }}
    />
  );
}

function NotebookImage({ src }: { src: string }) {
  return (
    <Image
      src={src}
      alt="Cell output"
      className="mt-1 max-w-full rounded"
      height={0}
      width={0}
      sizes="100vw"
      unoptimized
      style={{ height: "auto", width: "auto" }}
    />
  );
}

function NotebookOutputBlock({ output }: { output: NotebookOutput }) {
  if (output.output_type === "error") {
    const traceback = (output.traceback ?? [])
      .map((line) => line.replace(/\x1b\[[0-9;]*m/g, ""))
      .join("\n");
    return (
      <div className="mt-1 rounded border border-red-800/40 bg-red-950/30 p-2 font-mono text-xs text-red-400">
        <div className="font-semibold">
          {output.ename}: {output.evalue}
        </div>
        {traceback && (
          <pre className="mt-1 whitespace-pre-wrap break-words opacity-80">
            {traceback}
          </pre>
        )}
      </div>
    );
  }

  if (output.output_type === "stream") {
    const text = joinSource(output.text ?? "");
    if (!text) return null;
    const isStderr = output.name === "stderr";
    return (
      <pre
        className={`mt-1 whitespace-pre-wrap break-words rounded p-2 font-mono text-xs ${
          isStderr
            ? "border border-yellow-800/40 bg-yellow-950/30 text-yellow-300"
            : "bg-muted text-muted-foreground"
        }`}
      >
        {text}
      </pre>
    );
  }

  if (
    output.output_type === "display_data" ||
    output.output_type === "execute_result"
  ) {
    const data = output.data ?? {};
    const png = data["image/png"];
    if (png) {
      return <NotebookImage src={`data:image/png;base64,${joinSource(png)}`} />;
    }

    const jpeg = data["image/jpeg"];
    if (jpeg) {
      return (
        <NotebookImage src={`data:image/jpeg;base64,${joinSource(jpeg)}`} />
      );
    }

    const svg = data["image/svg+xml"];
    if (svg) {
      return (
        <SanitizedNotebookMarkup className="mt-1" markup={joinSource(svg)} />
      );
    }

    const html = data["text/html"];
    if (html) {
      return (
        <SanitizedNotebookMarkup
          className="mt-1 overflow-x-auto rounded bg-muted p-2 text-sm"
          markup={joinSource(html)}
        />
      );
    }

    const plain = data["text/plain"];
    if (plain) {
      return (
        <pre className="mt-1 whitespace-pre-wrap break-words rounded bg-muted p-2 font-mono text-xs text-muted-foreground">
          {joinSource(plain)}
        </pre>
      );
    }
  }

  return null;
}

function CollapsibleOutputs({ outputs }: { outputs: NotebookOutput[] }) {
  const [open, setOpen] = useState(true);
  if (!outputs.length) return null;

  return (
    <div className="ml-4 border-l-2 border-muted pl-3">
      <button
        onClick={() => setOpen((value) => !value)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
      >
        {open ? (
          <CaretDown className="size-3" />
        ) : (
          <CaretRight className="size-3" />
        )}
        {open ? "Hide" : "Show"} output
      </button>
      {open && (
        <div className="mt-1 flex flex-col gap-1">
          {outputs.map((output, index) => (
            <NotebookOutputBlock key={index} output={output} />
          ))}
        </div>
      )}
    </div>
  );
}

function CodeCell({
  cell,
  language,
}: {
  cell: NotebookCell;
  language: string;
}) {
  const source = joinSource(cell.source);
  const execCount = cell.execution_count;
  const highlightLanguage = getHighlightLanguage(language);

  return (
    <div className="group flex gap-2">
      <div className="w-10 shrink-0 select-none pt-2 text-right font-mono text-xs text-muted-foreground">
        {execCount != null ? `[${execCount}]:` : "[ ]:"}
      </div>

      <div className="min-w-0 flex-1">
        <div className="relative">
          <div className="absolute right-2 top-1.5 z-10 rounded bg-background/80 px-1.5 py-0.5 text-xs text-muted-foreground">
            {language}
          </div>
          <pre className="overflow-x-auto rounded bg-muted p-3 pr-16">
            <code className={`language-${highlightLanguage} text-sm`}>
              {source}
            </code>
          </pre>
        </div>

        {cell.outputs && cell.outputs.length > 0 && (
          <CollapsibleOutputs outputs={cell.outputs} />
        )}
      </div>
    </div>
  );
}

function MarkdownCell({ cell }: { cell: NotebookCell }) {
  const source = joinSource(cell.source);
  return (
    <div className="px-2 py-1">
      <ReactMarkdown
        className="prose prose-sm dark:prose-invert max-w-none"
        remarkPlugins={[
          remarkGfm,
          [remarkMath, { singleDollarTextMath: false }],
        ]}
        rehypePlugins={[[rehypeKatex, { strict: false }], rehypeHighlight]}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}

function RawCell({ cell }: { cell: NotebookCell }) {
  const source = joinSource(cell.source);
  if (!source.trim()) return null;
  return (
    <pre className="whitespace-pre-wrap break-words px-2 py-1 font-mono text-xs text-muted-foreground">
      {source}
    </pre>
  );
}

function getHighlightLanguage(language: string): string {
  const normalized = language.toLowerCase().replace(/[^a-z0-9_-]/g, "");
  return normalized || "plaintext";
}

export function NotebookViewer({ notebook }: { notebook: Notebook }) {
  const language =
    notebook.metadata?.kernelspec?.language ??
    notebook.metadata?.language_info?.name ??
    "python";

  const version = notebook.metadata?.language_info?.version;

  return (
    <div className="flex flex-col gap-2 rounded-md border border-border bg-background p-3">
      <div className="flex items-center gap-2 border-b border-border pb-2 text-xs text-muted-foreground">
        <span className="rounded bg-muted px-2 py-0.5 font-mono font-medium capitalize">
          {language}
          {version ? ` ${version}` : ""}
        </span>
        <span>nbformat {notebook.nbformat}</span>
        <span>-</span>
        <span>{notebook.cells.length} cells</span>
      </div>

      {notebook.cells.map((cell, index) => (
        <div
          key={index}
          className="rounded border border-transparent transition-colors hover:border-border/50"
        >
          {cell.cell_type === "code" && (
            <CodeCell cell={cell} language={language} />
          )}
          {cell.cell_type === "markdown" && <MarkdownCell cell={cell} />}
          {cell.cell_type === "raw" && <RawCell cell={cell} />}
        </div>
      ))}
    </div>
  );
}
