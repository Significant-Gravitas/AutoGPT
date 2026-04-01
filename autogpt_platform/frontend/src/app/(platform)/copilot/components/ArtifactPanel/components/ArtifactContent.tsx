"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { globalRegistry } from "@/components/contextual/OutputRenderers";
import { codeRenderer } from "@/components/contextual/OutputRenderers/renderers/CodeRenderer";
import type { ArtifactRef } from "../../../store";
import { classifyArtifact } from "../helpers";
import { Suspense, useEffect, useRef, useState } from "react";
import { ArtifactReactPreview } from "./ArtifactReactPreview";

interface Props {
  artifact: ArtifactRef;
  isSourceView: boolean;
}

function ArtifactContentLoader({ artifact, isSourceView }: Props) {
  const [content, setContent] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollPositions = useRef(new Map<string, number>());
  const scrollRef = useRef<HTMLDivElement>(null);

  const classification = classifyArtifact(
    artifact.mimeType,
    artifact.title,
    artifact.sizeBytes,
  );

  // Save scroll position when switching artifacts
  useEffect(() => {
    return () => {
      if (scrollRef.current) {
        scrollPositions.current.set(artifact.id, scrollRef.current.scrollTop);
      }
    };
  }, [artifact.id]);

  // Restore scroll position
  useEffect(() => {
    const saved = scrollPositions.current.get(artifact.id);
    if (saved && scrollRef.current) {
      scrollRef.current.scrollTop = saved;
    }
  }, [artifact.id]);

  useEffect(() => {
    if (classification.type === "image") {
      setContent(null);
      setPdfUrl(null);
      setError(null);
      setIsLoading(false);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    setError(null);

    if (classification.type === "pdf") {
      let objectUrl: string | null = null;

      setContent(null);
      setPdfUrl(null);
      fetch(artifact.sourceUrl)
        .then((res) => {
          if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
          return res.blob();
        })
        .then((blob) => {
          objectUrl = URL.createObjectURL(blob);
          if (cancelled) {
            URL.revokeObjectURL(objectUrl);
            objectUrl = null;
            return;
          }
          setPdfUrl(objectUrl);
          setIsLoading(false);
        })
        .catch((err) => {
          if (!cancelled) {
            setError(err.message);
            setIsLoading(false);
          }
        });

      return () => {
        cancelled = true;
        if (objectUrl) {
          URL.revokeObjectURL(objectUrl);
        }
      };
    }

    setPdfUrl(null);
    fetch(artifact.sourceUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
        return res.text();
      })
      .then((text) => {
        if (!cancelled) {
          setContent(text);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [artifact.id, artifact.sourceUrl, classification.type]);

  if (isLoading) {
    return (
      <div className="space-y-3 p-4">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
        <Skeleton className="h-4 w-5/6" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 p-8 text-center">
        <p className="text-sm text-zinc-500">Failed to load content</p>
        <p className="text-xs text-zinc-400">{error}</p>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto">
      <ArtifactRenderer
        artifact={artifact}
        content={content}
        pdfUrl={pdfUrl}
        isSourceView={isSourceView}
        classification={classification}
      />
    </div>
  );
}

function ArtifactRenderer({
  artifact,
  content,
  pdfUrl,
  isSourceView,
  classification,
}: {
  artifact: ArtifactRef;
  content: string | null;
  pdfUrl: string | null;
  isSourceView: boolean;
  classification: ReturnType<typeof classifyArtifact>;
}) {
  // Image: render directly from URL (no content fetch)
  if (classification.type === "image") {
    return (
      <div className="flex items-center justify-center p-4">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={artifact.sourceUrl}
          alt={artifact.title}
          className="max-h-full max-w-full object-contain"
        />
      </div>
    );
  }

  if (classification.type === "pdf" && pdfUrl) {
    return (
      <iframe src={pdfUrl} className="h-full w-full" title={artifact.title} />
    );
  }

  if (content === null) return null;

  // Source view: always show raw text
  if (isSourceView) {
    return (
      <pre className="whitespace-pre-wrap break-words p-4 font-mono text-sm text-zinc-800">
        {content}
      </pre>
    );
  }

  if (classification.type === "html") {
    return (
      <iframe
        sandbox="allow-scripts"
        srcDoc={content}
        className="h-full w-full border-0"
        title={artifact.title}
      />
    );
  }

  if (classification.type === "react") {
    return <ArtifactReactPreview source={content} title={artifact.title} />;
  }

  // Code: pass with explicit type metadata so CodeRenderer matches
  // (prevents higher-priority MarkdownRenderer from claiming it)
  if (classification.type === "code") {
    const ext = artifact.title.split(".").pop() ?? "";
    const codeMeta = {
      mimeType: artifact.mimeType ?? undefined,
      filename: artifact.title,
      type: "code",
      language: ext,
    };
    return <div className="p-4">{codeRenderer.render(content, codeMeta)}</div>;
  }

  // JSON: parse first so the JSONRenderer gets an object, not a string
  // (prevents higher-priority MarkdownRenderer from claiming it)
  if (classification.type === "json") {
    try {
      const parsed = JSON.parse(content);
      const jsonMeta = {
        mimeType: "application/json",
        type: "json",
        filename: artifact.title,
      };
      const jsonRenderer = globalRegistry.getRenderer(parsed, jsonMeta);
      if (jsonRenderer) {
        return (
          <div className="p-4">{jsonRenderer.render(parsed, jsonMeta)}</div>
        );
      }
    } catch {
      // invalid JSON — fall through to plain text
    }
  }

  // CSV: pass with explicit metadata so CSVRenderer matches
  if (classification.type === "csv") {
    const csvMeta = { mimeType: "text/csv", filename: artifact.title };
    const csvRenderer = globalRegistry.getRenderer(content, csvMeta);
    if (csvRenderer) {
      return <div className="p-4">{csvRenderer.render(content, csvMeta)}</div>;
    }
  }

  // Try the global renderer registry
  const metadata = {
    mimeType: artifact.mimeType ?? undefined,
    filename: artifact.title,
  };
  const renderer = globalRegistry.getRenderer(content, metadata);
  if (renderer) {
    return <div className="p-4">{renderer.render(content, metadata)}</div>;
  }

  // Fallback: plain text
  return (
    <pre className="whitespace-pre-wrap break-words p-4 font-mono text-sm text-zinc-800">
      {content}
    </pre>
  );
}

export function ArtifactContent(props: Props) {
  return (
    <Suspense
      fallback={
        <div className="space-y-3 p-4">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-32 w-full" />
        </div>
      }
    >
      <ArtifactContentLoader {...props} />
    </Suspense>
  );
}
