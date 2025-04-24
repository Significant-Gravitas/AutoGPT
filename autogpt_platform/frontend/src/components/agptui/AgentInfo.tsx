"use client";

import * as React from "react";
import { IconPlay, StarRatingIcons } from "@/components/ui/icons";
import { Separator } from "@/components/ui/separator";
import BackendAPI from "@/lib/autogpt-server-api";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useToast } from "@/components/ui/use-toast";

import useSupabase from "@/hooks/useSupabase";
import { DownloadIcon, LoaderIcon } from "lucide-react";
import { useOnboarding } from "../onboarding/onboarding-provider";
interface AgentInfoProps {
  name: string;
  creator: string;
  shortDescription: string;
  longDescription: string;
  rating: number;
  runs: number;
  categories: string[];
  lastUpdated: string;
  version: string;
  storeListingVersionId: string;
}

export const AgentInfo: React.FC<AgentInfoProps> = ({
  name,
  creator,
  shortDescription,
  longDescription,
  rating,
  runs,
  categories,
  lastUpdated,
  version,
  storeListingVersionId,
}) => {
  const router = useRouter();
  const api = React.useMemo(() => new BackendAPI(), []);
  const { user } = useSupabase();
  const { toast } = useToast();
  const { completeStep } = useOnboarding();

  const [downloading, setDownloading] = React.useState(false);

  const handleAddToLibrary = async () => {
    try {
      const newLibraryAgent = await api.addMarketplaceAgentToLibrary(
        storeListingVersionId,
      );
      completeStep("MARKETPLACE_ADD_AGENT");
      router.push(`/library/agents/${newLibraryAgent.id}`);
    } catch (error) {
      console.error("Failed to add agent to library:", error);
    }
  };

  const handleDownloadToLibrary = async () => {
    const downloadAgent = async (): Promise<void> => {
      setDownloading(true);
      try {
        const file = await api.downloadStoreAgent(storeListingVersionId);

        // Similar to Marketplace v1
        const jsonData = JSON.stringify(file, null, 2);
        // Create a Blob from the file content
        const blob = new Blob([jsonData], { type: "application/json" });

        // Create a temporary URL for the Blob
        const url = window.URL.createObjectURL(blob);

        // Create a temporary anchor element
        const a = document.createElement("a");
        a.href = url;
        a.download = `agent_${storeListingVersionId}.json`; // Set the filename

        // Append the anchor to the body, click it, and remove it
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // Revoke the temporary URL
        window.URL.revokeObjectURL(url);

        toast({
          title: "Download Complete",
          description: "Your agent has been successfully downloaded.",
        });
      } catch (error) {
        console.error(`Error downloading agent:`, error);
        throw error;
      }
    };
    await downloadAgent();
    setDownloading(false);
  };

  return (
    <div className="w-full max-w-[25rem] space-y-11">
      {/* Top part */}
      <div>
        <h2 className="mb-3 font-poppins text-4xl font-medium text-neutral-900">
          {name}
        </h2>
        <div className="mb-7 flex w-full items-center gap-1.5 font-sans">
          <p className="text-xl font-normal text-neutral-800">by</p>
          <Link
            href={`/marketplace/creator/${encodeURIComponent(creator)}`}
            className="text-xl font-medium text-neutral-800 hover:underline"
          >
            {creator}
          </Link>
        </div>

        <p className="mb-7 line-clamp-2 font-sans text-xl font-normal text-neutral-600">
          {shortDescription}
        </p>

        <div className="mb-14 w-full">
          {user ? (
            <button
              onClick={handleAddToLibrary}
              className="inline-flex w-full items-center justify-center gap-2 rounded-[38px] bg-violet-600 px-4 py-3 transition-colors hover:bg-violet-700 sm:w-auto sm:gap-2.5 sm:px-5 sm:py-3.5 lg:px-6 lg:py-4"
            >
              <IconPlay className="h-5 w-5 text-white sm:h-5 sm:w-5 lg:h-6 lg:w-6" />
              <span className="font-poppins text-base font-medium text-neutral-50 sm:text-lg">
                Add To Library
              </span>
            </button>
          ) : (
            <button
              onClick={handleDownloadToLibrary}
              className={`inline-flex w-full items-center justify-center gap-2 rounded-[38px] px-4 py-3 transition-colors sm:w-auto sm:gap-2.5 sm:px-5 sm:py-3.5 lg:px-6 lg:py-4 ${
                downloading
                  ? "bg-neutral-400"
                  : "bg-violet-600 hover:bg-violet-700"
              }`}
              disabled={downloading}
            >
              {downloading ? (
                <LoaderIcon className="h-5 w-5 animate-spin text-white sm:h-5 sm:w-5 lg:h-6 lg:w-6" />
              ) : (
                <DownloadIcon className="h-5 w-5 text-white sm:h-5 sm:w-5 lg:h-6 lg:w-6" />
              )}
              <span className="font-poppins text-base font-medium text-neutral-50 sm:text-lg">
                {downloading ? "Downloading..." : "Download Agent as File"}
              </span>
            </button>
          )}
        </div>

        <div className="flex w-full items-center justify-between">
          <div className="flex items-center gap-1.5 sm:gap-2">
            <span className="font-sans text-lg font-semibold text-neutral-800">
              {rating.toFixed(1)}
            </span>
            <div className="flex gap-0.5">{StarRatingIcons(rating)}</div>
          </div>
          <div className="font-sans text-lg font-semibold text-neutral-800">
            {runs.toLocaleString()} runs
          </div>
        </div>
      </div>

      {/* Separator */}
      <Separator className="mb-4 lg:mb-[44px]" />

      {/* Bottom part */}
      <div className="space-y-9">
        <div className="space-y-2.5">
          <p className="font-sans text-base font-medium text-neutral-800">
            Description
          </p>
          <p className="whitespace-pre-line font-sans text-base font-normal text-neutral-600">
            {longDescription}
          </p>
        </div>

        {/* Categories */}
        <div className="space-y-2.5">
          <p className="font-sans text-base font-medium text-neutral-800">
            Categories
          </p>
          <div className="flex flex-wrap gap-2.5">
            {categories.map((category, index) => (
              <p
                key={index}
                className="rounded-full border border-neutral-600 bg-white px-5 py-3 font-sans text-base font-normal text-neutral-800"
              >
                {category}
              </p>
            ))}
          </div>
        </div>

        {/* TODO : Rating Agent */}

        {/* Version History */}
        <div className="flex flex-col gap-2.5">
          <p className="font-base font-sans font-medium text-neutral-800">
            Version history
          </p>
          <p className="font-sans text-base font-normal text-neutral-600">
            Last updated {lastUpdated}
          </p>
          <p className="font-sans text-base font-normal text-neutral-600">
            Version {version}
          </p>
        </div>
      </div>
    </div>
  );
};
