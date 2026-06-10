"use client";

import { useEffect } from "react";
import { notFound } from "next/navigation";
import { motion, useReducedMotion } from "framer-motion";
import type { Transition, Variants } from "framer-motion";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { ArtifactsSearchBar } from "./components/ArtifactsSearchBar/ArtifactsSearchBar";
import { ArtifactsList } from "./components/ArtifactsList/ArtifactsList";
import { OriginFilter } from "./components/OriginFilter/OriginFilter";
import { StorageUsage } from "./components/StorageUsage/StorageUsage";
import { useArtifactsPage } from "./useArtifactsPage";

const EASE_OUT_SOFT: Transition["ease"] = [0.16, 1, 0.3, 1];

const SECTION_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 8, filter: "blur(8px)" },
  show: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.45, ease: EASE_OUT_SOFT },
  },
};

const REDUCED_SECTION_VARIANTS: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0.2 } },
};

export default function ArtifactsPage() {
  const { enabled: isEnabled, ready: flagReady } = useFlagStatus(
    Flag.ARTIFACTS_PAGE,
  );
  const reduceMotion = useReducedMotion();
  const {
    files,
    isLoading,
    isError,
    error,
    searchTerm,
    setSearchTerm,
    debouncedSearch,
    originFilter,
    setOriginFilter,
    hasMore,
    isLoadingMore,
    loadMore,
  } = useArtifactsPage();

  useEffect(() => {
    document.title = "Files – AutoGPT Platform";
  }, []);

  if (!flagReady) {
    return <ArtifactsPageSkeleton />;
  }
  if (!isEnabled) {
    notFound();
  }

  const variants = reduceMotion ? REDUCED_SECTION_VARIANTS : SECTION_VARIANTS;

  return (
    <main className="container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <motion.div
          className="flex flex-col gap-1"
          variants={variants}
          initial="hidden"
          animate="show"
          transition={{ delay: 0 }}
        >
          <Text variant="h3">Files</Text>
          <Text variant="body" className="max-w-prose text-zinc-600">
            Every file your agents generate or use in the Builder lives here —
            ready to reuse, download, or share.
          </Text>
        </motion.div>
        <motion.div
          variants={variants}
          initial="hidden"
          animate="show"
          transition={{ delay: reduceMotion ? 0 : 0.08 }}
        >
          <ArtifactsSearchBar
            searchTerm={searchTerm}
            setSearchTerm={setSearchTerm}
          />
        </motion.div>
      </div>
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <motion.div
          className="flex w-full md:w-2/5"
          variants={variants}
          initial="hidden"
          animate="show"
          transition={{ delay: reduceMotion ? 0 : 0.16 }}
        >
          <StorageUsage />
        </motion.div>
        <motion.div
          variants={variants}
          initial="hidden"
          animate="show"
          transition={{ delay: reduceMotion ? 0 : 0.24 }}
        >
          <OriginFilter value={originFilter} onChange={setOriginFilter} />
        </motion.div>
      </div>
      <motion.div
        variants={variants}
        initial="hidden"
        animate="show"
        transition={{ delay: reduceMotion ? 0 : 0.32 }}
      >
        <ArtifactsList
          files={files}
          isLoading={isLoading}
          isError={isError}
          error={error}
          hasSearchTerm={searchTerm.length > 0}
          hasMore={hasMore}
          isLoadingMore={isLoadingMore}
          onLoadMore={loadMore}
          listKey={`${originFilter}|${debouncedSearch}`}
        />
      </motion.div>
    </main>
  );
}

function ArtifactsPageSkeleton() {
  return (
    <main
      className="container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12"
      data-testid="artifacts-flag-loading"
    >
      <Skeleton className="h-8 w-48 rounded-md" />
      <Skeleton className="h-4 w-80 rounded-md" />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-64 w-full rounded-2xl" />
        ))}
      </div>
    </main>
  );
}
