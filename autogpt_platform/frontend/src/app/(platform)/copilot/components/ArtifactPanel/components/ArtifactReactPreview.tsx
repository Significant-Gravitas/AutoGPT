"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { useEffect, useState } from "react";
import {
  buildReactArtifactSrcDoc,
  collectPreviewStyles,
  transpileReactArtifactSource,
} from "./reactArtifactPreview";

interface Props {
  source: string;
  title: string;
}

export function ArtifactReactPreview({ source, title }: Props) {
  const [srcDoc, setSrcDoc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    setSrcDoc(null);
    setError(null);

    transpileReactArtifactSource(source, title)
      .then((compiledCode) => {
        if (cancelled) return;
        setSrcDoc(
          buildReactArtifactSrcDoc(compiledCode, title, collectPreviewStyles()),
        );
      })
      .catch((nextError: unknown) => {
        if (cancelled) return;
        setError(
          nextError instanceof Error
            ? nextError.message
            : "Failed to build artifact preview",
        );
      });

    return () => {
      cancelled = true;
    };
  }, [source, title]);

  if (error) {
    return (
      <div className="flex flex-col gap-2 p-4">
        <p className="text-sm font-medium text-red-600">
          Failed to render React preview
        </p>
        <pre className="whitespace-pre-wrap break-words rounded-md bg-red-50 p-3 font-mono text-xs text-red-900">
          {error}
        </pre>
      </div>
    );
  }

  if (!srcDoc) {
    return (
      <div className="space-y-3 p-4">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  return (
    <iframe
      sandbox="allow-scripts"
      srcDoc={srcDoc}
      className="h-full w-full border-0"
      title={`${title} preview`}
    />
  );
}
