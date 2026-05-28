"use client";

import { useEffect } from "react";
import { notFound } from "next/navigation";
import { Text } from "@/components/atoms/Text/Text";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ArtifactsSearchBar } from "./components/ArtifactsSearchBar/ArtifactsSearchBar";
import { ArtifactsList } from "./components/ArtifactsList/ArtifactsList";
import { OriginFilter } from "./components/OriginFilter/OriginFilter";
import { StorageUsage } from "./components/StorageUsage/StorageUsage";
import { useArtifactsPage } from "./useArtifactsPage";

export default function ArtifactsPage() {
  const isEnabled = useGetFlag(Flag.ARTIFACTS_PAGE);
  const {
    files,
    isLoading,
    isError,
    error,
    searchTerm,
    setSearchTerm,
    originFilter,
    setOriginFilter,
  } = useArtifactsPage();

  useEffect(() => {
    document.title = "Artifacts – AutoGPT Platform";
  }, []);

  if (!isEnabled) {
    notFound();
  }

  return (
    <main className="container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="flex flex-col gap-1">
          <Text variant="h3">Artifacts</Text>
          <Text variant="body" className="text-zinc-600">
            Every file your agents generate or use in the Builder lives here{" "}
            <br />
            ready to reuse, download, or share.
          </Text>
        </div>
        <ArtifactsSearchBar
          searchTerm={searchTerm}
          setSearchTerm={setSearchTerm}
        />
      </div>
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <StorageUsage />
        <OriginFilter value={originFilter} onChange={setOriginFilter} />
      </div>
      <ArtifactsList
        files={files}
        isLoading={isLoading}
        isError={isError}
        error={error}
        hasSearchTerm={searchTerm.length > 0}
      />
    </main>
  );
}
