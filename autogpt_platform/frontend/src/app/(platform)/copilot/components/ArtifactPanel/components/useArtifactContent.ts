"use client";

import { useEffect, useRef, useState } from "react";
import type { ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";

interface Result {
  content: string | null;
  pdfUrl: string | null;
  isLoading: boolean;
  error: string | null;
  scrollRef: React.RefObject<HTMLDivElement>;
}

export function useArtifactContent(
  artifact: ArtifactRef,
  classification: ArtifactClassification,
): Result {
  const [content, setContent] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollPositions = useRef(new Map<string, number>());
  const scrollRef = useRef<HTMLDivElement>(null);

  // Save scroll position when switching artifacts. Only save when the
  // content div has actually been mounted with a nonzero scrollTop, so we
  // don't overwrite a previously-saved position with 0 from a skeleton render.
  useEffect(() => {
    return () => {
      const node = scrollRef.current;
      if (node && node.scrollTop > 0) {
        scrollPositions.current.set(artifact.id, node.scrollTop);
      }
    };
  }, [artifact.id]);

  // Restore scroll position — wait until isLoading flips to false, since
  // the scroll container is replaced by a Skeleton during loading and the
  // real content div would otherwise mount with scrollTop=0.
  useEffect(() => {
    if (isLoading) return;
    const saved = scrollPositions.current.get(artifact.id);
    if (saved != null && scrollRef.current) {
      scrollRef.current.scrollTop = saved;
    }
  }, [artifact.id, isLoading]);

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
        if (objectUrl) URL.revokeObjectURL(objectUrl);
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

  return { content, pdfUrl, isLoading, error, scrollRef };
}
