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
import AutogptButton from "./AutogptButton";
import { Chip } from "./Chip";
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
    <div className="w-full max-w-[27rem] space-y-7">
      {/* Top part */}
      <div className="space-y-14">
        {/* Agent name */}
        <div>
          <h2 className="font-poppins text-[1.75rem] font-medium leading-[2.5rem] text-zinc-800">
            {name}
          </h2>

          {/* Creator name */}
          <div className="mb-7 flex w-full items-center gap-1.5 font-sans">
            <p className="text-base font-normal text-zinc-800">by</p>
            <Link
              href={`/marketplace/creator/${encodeURIComponent(creator)}`}
              className="text-base font-medium text-zinc-800 hover:underline"
            >
              {creator}
            </Link>
          </div>
        </div>

        {/* Download and run button */}
        <div className="w-full">
          {user ? (
            <AutogptButton variant={"secondary"} onClick={handleAddToLibrary}>
              Add To Library
            </AutogptButton>
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
        {/* Runs and ratings */}
        <div className="flex w-full items-center gap-10">
          <div className="flex items-center gap-1.5 sm:gap-2">
            <span className="font-sans text-base font-medium text-zinc-800">
              {rating.toFixed(1)}
            </span>
            <div className="flex gap-0.5">{StarRatingIcons(rating)}</div>
          </div>
          <div className="font-sans text-base font-medium text-zinc-800">
            {runs.toLocaleString()} runs
          </div>
        </div>
      </div>

      {/* Separator */}
      <Separator className="bg-neutral-300" />

      {/* Bottom part */}
      <div className="space-y-9">
        <div className="space-y-2.5">
          <p className="font-sans text-base font-medium text-zinc-800">
            Description
          </p>
          <p className="whitespace-pre-line font-sans text-base font-normal text-zinc-600">
            {longDescription}
          </p>
        </div>

        {/* Categories */}
        <div className="space-y-2.5">
          <p className="font-sans text-base font-medium text-zinc-800">
            Categories
          </p>
          <div className="flex flex-wrap gap-2.5">
            {categories.map((category, index) => (
              <Chip key={index}>{category}</Chip>
            ))}
          </div>
        </div>

        {/* TODO : Rating Agent */}

        {/* Version History */}
        <div className="space-y-2.5">
          <p className="font-sans text-base font-medium text-zinc-800">
            Version history
          </p>
          <div className="space-y-1.5">
            <p className="font-sans text-base font-normal text-zinc-600">
              Last updated {lastUpdated}
            </p>
            <p className="font-sans text-base font-normal text-zinc-600">
              Version {version}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
