"use client";

import { Text } from "@/components/atoms/Text/Text";
import { useEffect, useState } from "react";
import { ArtifactSkeleton } from "./ArtifactSkeleton";
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
        <Text variant="body-medium" as="p" tone="danger">
          Failed to render React preview
        </Text>
        <pre className="rounded-md bg-red-50 p-3 font-mono text-xs wrap-break-word whitespace-pre-wrap text-red-900">
          {error}
        </pre>
      </div>
    );
  }

  if (!srcDoc) {
    return <ArtifactSkeleton />;
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
