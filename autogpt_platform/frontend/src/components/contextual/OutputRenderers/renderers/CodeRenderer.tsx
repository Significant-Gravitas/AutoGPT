"use client";

import React, { useEffect, useState } from "react";
import {
  SHIKI_THEMES,
  type BundledLanguage,
  getShikiHighlighter,
  isLanguageSupported,
  resolveLanguage,
} from "@/lib/shiki-highlighter";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

interface HighlightToken {
  content: string;
  color?: string;
  htmlStyle?: Record<string, string>;
}

interface HighlightedCodeState {
  tokens: HighlightToken[][];
  fg?: string;
  bg?: string;
}

function getFileExtension(language: string): string {
  const extensionMap: Record<string, string> = {
    javascript: "js",
    typescript: "ts",
    python: "py",
    java: "java",
    csharp: "cs",
    cpp: "cpp",
    c: "c",
    html: "html",
    css: "css",
    json: "json",
    xml: "xml",
    yaml: "yaml",
    markdown: "md",
    sql: "sql",
    bash: "sh",
    shell: "sh",
    plaintext: "txt",
  };

  return extensionMap[language.toLowerCase()] || "txt";
}

function canRenderCode(value: unknown, metadata?: OutputMetadata): boolean {
  if (metadata?.type === "code" || metadata?.language) {
    return typeof value === "string";
  }

  if (typeof value !== "string") return false;

  const markdownIndicators = [
    /^#{1,6}\s+/m,
    /\*\*[^*]+\*\*/,
    /\[([^\]]+)\]\(([^)]+)\)/,
    /^>\s+/m,
    /^\s*[-*+]\s+\w+/m,
    /!\[([^\]]*)\]\(([^)]+)\)/,
  ];

  let markdownMatches = 0;
  for (const pattern of markdownIndicators) {
    if (pattern.test(value)) {
      markdownMatches++;
      if (markdownMatches >= 2) {
        return false;
      }
    }
  }

  const codeIndicators = [
    /^(function|const|let|var|class|import|export|if|for|while)\s/m,
    /^def\s+\w+\s*\(/m,
    /^import\s+/m,
    /^from\s+\w+\s+import/m,
    /^\s*<[^>]+>/,
    /[{}[\]();]/,
  ];

  return codeIndicators.some((pattern) => pattern.test(value));
}

function EditorLineNumber({ index }: { index: number }) {
  return (
    <span className="select-none pr-2 text-right font-mono text-xs text-zinc-600">
      {index + 1}
    </span>
  );
}

function PlainCodeLines({ code }: { code: string }) {
  return code.split("\n").map((line, index) => (
    <div key={`${index}-${line}`} className="grid grid-cols-[3rem_1fr] gap-4">
      <EditorLineNumber index={index} />
      <span className="whitespace-pre font-mono text-sm text-zinc-100">
        {line || " "}
      </span>
    </div>
  ));
}

function HighlightedCodeBlock({
  code,
  filename,
  language,
}: {
  code: string;
  filename?: string;
  language?: string;
}) {
  const [highlighted, setHighlighted] = useState<HighlightedCodeState | null>(
    null,
  );
  const resolvedLanguage = resolveLanguage(language || "text");
  const supportedLanguage = isLanguageSupported(resolvedLanguage)
    ? resolvedLanguage
    : "text";

  useEffect(() => {
    let cancelled = false;
    const shikiLanguage = supportedLanguage as BundledLanguage;

    setHighlighted(null);

    getShikiHighlighter()
      .then(async (highlighter) => {
        if (
          supportedLanguage !== "text" &&
          !highlighter.getLoadedLanguages().includes(supportedLanguage)
        ) {
          await highlighter.loadLanguage(shikiLanguage);
        }

        const shikiResult = highlighter.codeToTokens(code, {
          lang: shikiLanguage,
          theme: SHIKI_THEMES[1],
        });

        if (cancelled) return;

        setHighlighted({
          tokens: shikiResult.tokens.map((line) =>
            line.map((token) => ({
              content: token.content,
              color: token.color,
              htmlStyle: token.htmlStyle,
            })),
          ),
          fg: shikiResult.fg,
          bg: shikiResult.bg,
        });
      })
      .catch(() => {
        if (cancelled) return;
        setHighlighted(null);
      });

    return () => {
      cancelled = true;
    };
  }, [code, supportedLanguage]);

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-900 bg-[#020617] shadow-sm">
      <div className="flex items-center justify-between border-b border-zinc-800 bg-[#111827] px-3 py-2">
        <span className="truncate font-mono text-xs text-zinc-400">
          {filename || "code"}
        </span>
        <span className="rounded bg-zinc-800 px-2 py-0.5 font-mono text-[11px] uppercase tracking-wide text-zinc-300">
          {supportedLanguage}
        </span>
      </div>
      <div
        className="overflow-x-auto"
        style={{
          backgroundColor: highlighted?.bg || "#020617",
          color: highlighted?.fg || "#e2e8f0",
        }}
      >
        <pre className="min-w-full p-4">
          {highlighted ? (
            highlighted.tokens.map((line, index) => (
              <div
                key={`${index}-${line.length}`}
                className="grid grid-cols-[3rem_1fr] gap-4"
              >
                <EditorLineNumber index={index} />
                <span className="whitespace-pre font-mono text-sm leading-6">
                  {line.length > 0
                    ? line.map((token, tokenIndex) => (
                        <span
                          key={`${index}-${tokenIndex}-${token.content}`}
                          style={
                            token.htmlStyle
                              ? (token.htmlStyle as React.CSSProperties)
                              : token.color
                                ? { color: token.color }
                                : undefined
                          }
                        >
                          {token.content}
                        </span>
                      ))
                    : " "}
                </span>
              </div>
            ))
          ) : (
            <PlainCodeLines code={code} />
          )}
        </pre>
      </div>
    </div>
  );
}

function renderCode(
  value: unknown,
  metadata?: OutputMetadata,
): React.ReactNode {
  const codeValue = String(value);
  const language = metadata?.language || "text";

  return (
    <HighlightedCodeBlock
      code={codeValue}
      filename={metadata?.filename}
      language={language}
    />
  );
}

function getCopyContentCode(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  const codeValue = String(value);
  return {
    mimeType: "text/plain",
    data: codeValue,
    fallbackText: codeValue,
  };
}

function getDownloadContentCode(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const codeValue = String(value);
  const language = metadata?.language || "txt";
  const extension = getFileExtension(language);
  const blob = new Blob([codeValue], { type: "text/plain" });

  return {
    data: blob,
    filename: metadata?.filename || `code.${extension}`,
    mimeType: "text/plain",
  };
}

function isConcatenableCode(
  _value: unknown,
  _metadata?: OutputMetadata,
): boolean {
  return true;
}

export const codeRenderer: OutputRenderer = {
  name: "CodeRenderer",
  priority: 30,
  canRender: canRenderCode,
  render: renderCode,
  getCopyContent: getCopyContentCode,
  getDownloadContent: getDownloadContentCode,
  isConcatenable: isConcatenableCode,
};
