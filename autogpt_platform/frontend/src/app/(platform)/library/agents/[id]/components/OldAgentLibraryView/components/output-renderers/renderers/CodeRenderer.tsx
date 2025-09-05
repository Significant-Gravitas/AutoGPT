import React from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

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

function renderCode(
  value: unknown,
  metadata?: OutputMetadata,
): React.ReactNode {
  const codeValue = String(value);
  const language = metadata?.language || "plaintext";

  return (
    <div className="group relative">
      {metadata?.language && (
        <div className="absolute right-2 top-2 rounded bg-background/80 px-2 py-1 text-xs text-muted-foreground">
          {language}
        </div>
      )}
      <pre className="overflow-x-auto rounded-md bg-muted p-3">
        <code className="font-mono text-sm">{codeValue}</code>
      </pre>
    </div>
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
