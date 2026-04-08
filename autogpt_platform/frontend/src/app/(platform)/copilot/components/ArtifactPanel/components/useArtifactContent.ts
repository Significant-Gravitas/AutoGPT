"use client";

import { useEffect, useRef, useState } from "react";
import type { ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";

// Cap on cached text artifacts. Long sessions with many large artifacts
// would otherwise hold every opened one in memory.
const CONTENT_CACHE_MAX = 12;

// Module-level LRU keyed by artifact id so a sibling action (e.g. Copy
// in ArtifactPanelHeader) can read what the panel already fetched without
// re-hitting the network.
const contentCache = new Map<string, string>();

export function getCachedArtifactContent(id: string): string | undefined {
  return contentCache.get(id);
}

export function clearContentCache() {
  contentCache.clear();
}

export function useArtifactContent(
  artifact: ArtifactRef,
  classification: ArtifactClassification,
) {
  const [content, setContent] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Bumped by `retry()` to force the fetch effect to re-run.
  const [retryNonce, setRetryNonce] = useState(0);
  const scrollPositions = useRef(new Map<string, number>());
  const scrollRef = useRef<HTMLDivElement>(null);

  function retry() {
    // Drop any cached failure/content for this id so we actually re-fetch.
    contentCache.delete(artifact.id);
    setRetryNonce((n) => n + 1);
  }

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
    // LRU touch — re-insert so the most-recently-used entry sits at the
    // tail and the oldest entry falls off the head first.
    const cache = contentCache;
    const cached = cache.get(artifact.id);
    if (cached !== undefined) {
      cache.delete(artifact.id);
      cache.set(artifact.id, cached);
      setContent(cached);
      setIsLoading(false);
      return () => {
        cancelled = true;
      };
    }
    fetch(artifact.sourceUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
        return res.text();
      })
      .then((text) => {
        if (!cancelled) {
          if (cache.size >= CONTENT_CACHE_MAX) {
            // Map preserves insertion order — first key is the oldest.
            const oldest = cache.keys().next().value;
            if (oldest !== undefined) cache.delete(oldest);
          }
          cache.set(artifact.id, text);
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
  }, [artifact.id, artifact.sourceUrl, classification.type, retryNonce]);

  return { content, pdfUrl, isLoading, error, scrollRef, retry };
}
