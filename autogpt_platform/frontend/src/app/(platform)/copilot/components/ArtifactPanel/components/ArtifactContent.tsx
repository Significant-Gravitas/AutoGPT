"use client";

import { Text } from "@/components/atoms/Text/Text";
import { globalRegistry } from "@/components/contextual/OutputRenderers";
import { codeRenderer } from "@/components/contextual/OutputRenderers/renderers/CodeRenderer";
import { Suspense, useState } from "react";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import type { ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";
import { ArtifactErrorBoundary } from "./ArtifactErrorBoundary";
import { ArtifactReactPreview } from "./ArtifactReactPreview";
import { ArtifactSkeleton } from "./ArtifactSkeleton";
import { ExpertArtifactContent } from "./ExpertArtifactContent";
import {
  FRAGMENT_LINK_INTERCEPTOR_SCRIPT,
  TAILWIND_CDN_URL,
  wrapWithHeadInjection,
} from "@/lib/iframe-sandbox-csp";
import { useArtifactContent } from "./useArtifactContent";

// Text is typed with <p> props; the retry buttons keep their type="button".
const BUTTON_TYPE = { type: "button" } as const;

interface Props {
  artifact: ArtifactRef;
  isSourceView: boolean;
  classification: ArtifactClassification;
}

function ArtifactContentLoader({
  artifact,
  isSourceView,
  classification,
}: Props) {
  const { content, pdfUrl, isLoading, error, scrollRef, retry } =
    useArtifactContent(artifact, classification);

  if (isLoading) {
    return <ArtifactSkeleton extraLine />;
  }

  if (error) {
    return (
      <div
        role="alert"
        className="flex flex-col items-center justify-center gap-3 p-8 text-center"
      >
        <Text variant="body-tight" as="p" tone="muted">
          Failed to load content
        </Text>
        <Text variant="small-tight" as="p" tone="subtle">
          {error}
        </Text>
        <Text
          as="button"
          variant="small-medium-tight"
          tone="body"
          {...BUTTON_TYPE}
          onClick={retry}
          // TODO(tokens): no exact role for focus-visible:ring-violet-400
          className="rounded-md border border-border-base bg-surface px-3 py-1.5 shadow-sm transition-colors hover:bg-surface-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
        >
          Try again
        </Text>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto">
      <ArtifactErrorBoundary
        artifactID={artifact.id}
        artifactTitle={artifact.title}
        artifactType={classification.type}
      >
        <ArtifactRenderer
          artifact={artifact}
          content={content}
          pdfUrl={pdfUrl}
          isSourceView={isSourceView}
          classification={classification}
        />
      </ArtifactErrorBoundary>
    </div>
  );
}

function withCacheBust(src: string, nonce: number): string {
  if (nonce === 0) return src;
  const sep = src.includes("?") ? "&" : "?";
  return `${src}${sep}_retry=${nonce}`;
}

function ArtifactImage({ src, alt }: { src: string; alt: string }) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  // Incremented on every Try Again so the URL changes and the browser
  // can't reuse a negative-cached response (SECRT-2221).
  const [retryNonce, setRetryNonce] = useState(0);

  if (error) {
    return (
      <div
        role="alert"
        className="flex flex-col items-center justify-center gap-3 p-8 text-center"
      >
        <Text variant="body-tight" as="p" tone="muted">
          Failed to load image
        </Text>
        <Text
          as="button"
          variant="small-medium-tight"
          tone="body"
          {...BUTTON_TYPE}
          onClick={() => {
            setError(false);
            setLoaded(false);
            setRetryNonce((n) => n + 1);
          }}
          // TODO(tokens): no exact role for focus-visible:ring-violet-400
          className="rounded-md border border-border-base bg-surface px-3 py-1.5 shadow-sm transition-colors hover:bg-surface-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
        >
          Try again
        </Text>
      </div>
    );
  }

  return (
    <div className="relative flex items-center justify-center p-4">
      {!loaded && (
        <Skeleton
          variant="translucent"
          className="absolute inset-4 h-[calc(100%-2rem)] w-[calc(100%-2rem)] rounded-md"
        />
      )}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={withCacheBust(src, retryNonce)}
        alt={alt}
        className={`max-h-full max-w-full object-contain transition-opacity ${loaded ? "opacity-100" : "opacity-0"}`}
        onLoad={() => setLoaded(true)}
        onError={() => setError(true)}
      />
    </div>
  );
}

function ArtifactVideo({ src }: { src: string }) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  const [retryNonce, setRetryNonce] = useState(0);

  if (error) {
    return (
      <div
        role="alert"
        className="flex flex-col items-center justify-center gap-3 p-8 text-center"
      >
        <Text variant="body-tight" as="p" tone="muted">
          Failed to load video
        </Text>
        <Text
          as="button"
          variant="small-medium-tight"
          tone="body"
          {...BUTTON_TYPE}
          onClick={() => {
            setError(false);
            setLoaded(false);
            setRetryNonce((n) => n + 1);
          }}
          // TODO(tokens): no exact role for focus-visible:ring-violet-400
          className="rounded-md border border-border-base bg-surface px-3 py-1.5 shadow-sm transition-colors hover:bg-surface-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
        >
          Try again
        </Text>
      </div>
    );
  }

  return (
    <div className="relative flex items-center justify-center p-4">
      {!loaded && (
        <Skeleton
          variant="translucent"
          className="absolute inset-4 h-[calc(100%-2rem)] w-[calc(100%-2rem)] rounded-md"
        />
      )}
      <video
        src={withCacheBust(src, retryNonce)}
        controls
        preload="metadata"
        className={`max-h-full max-w-full rounded-md transition-opacity ${loaded ? "opacity-100" : "opacity-0"}`}
        onLoadedMetadata={() => setLoaded(true)}
        onError={() => setError(true)}
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
  classification: ArtifactClassification;
}) {
  // Image: render directly from URL (no content fetch)
  if (classification.type === "image") {
    return (
      <ArtifactImage
        key={artifact.sourceUrl}
        src={artifact.sourceUrl}
        alt={artifact.title}
      />
    );
  }

  // Video: render with <video> controls (no content fetch)
  if (classification.type === "video") {
    return <ArtifactVideo key={artifact.sourceUrl} src={artifact.sourceUrl} />;
  }

  if (classification.type === "pdf" && pdfUrl) {
    // No sandbox — Chrome/Edge block PDF rendering in sandboxed iframes
    // (Chromium bug #413851). The blob URL has a null origin so it can't
    // access the parent page regardless.
    return (
      <iframe src={pdfUrl} className="h-full w-full" title={artifact.title} />
    );
  }

  if (content === null) return null;

  // Source view: always show raw text
  if (isSourceView) {
    return (
      <Text
        variant="body-mono-tight"
        as="pre"
        tone="emphasis"
        className="whitespace-pre-wrap break-words p-4"
      >
        {content}
      </Text>
    );
  }

  if (classification.type === "html") {
    // Inject Tailwind CDN — no CSP (see iframe-sandbox-csp.ts for why)
    const tailwindScript = `<script src="${TAILWIND_CDN_URL}"></script>`;
    const wrapped = wrapWithHeadInjection(
      content,
      tailwindScript + FRAGMENT_LINK_INTERCEPTOR_SCRIPT,
    );
    return (
      <iframe
        sandbox="allow-scripts"
        srcDoc={wrapped}
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
    const normalizedMime = artifact.mimeType
      ?.toLowerCase()
      .split(";")[0]
      ?.trim();
    const csvMimeType =
      normalizedMime === "text/tab-separated-values" ||
      artifact.title.toLowerCase().endsWith(".tsv")
        ? "text/tab-separated-values"
        : "text/csv";
    const csvMeta = { mimeType: csvMimeType, filename: artifact.title };
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
    <Text
      variant="body-mono-tight"
      as="pre"
      tone="emphasis"
      className="whitespace-pre-wrap break-words p-4"
    >
      {content}
    </Text>
  );
}

export function ArtifactContent(props: Props) {
  if (props.artifact.expert) {
    return <ExpertArtifactContent expert={props.artifact.expert} />;
  }
  return (
    <Suspense fallback={<ArtifactSkeleton />}>
      <ArtifactContentLoader {...props} />
    </Suspense>
  );
}
