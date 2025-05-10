"use client";

import { StarRatingIcons } from "@/components/ui/icons";
import { Separator } from "@/components/ui/separator";
import BackendAPI, { LibraryAgent } from "@/lib/autogpt-server-api";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useToast } from "@/components/ui/use-toast";

import { useOnboarding } from "../onboarding/onboarding-provider";
import AutogptButton from "./AutogptButton";
import { Chip } from "./Chip";
import { User } from "@supabase/supabase-js";
import { cn } from "@/lib/utils";
import { FC, useCallback, useMemo, useState } from "react";

interface AgentInfoProps {
  user: User | null;
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
  libraryAgent: LibraryAgent | null;
}

export const AgentInfo: FC<AgentInfoProps> = ({
  user,
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
  libraryAgent,
}) => {
  const router = useRouter();
  const api = useMemo(() => new BackendAPI(), []);
  const { toast } = useToast();
  const { completeStep } = useOnboarding();
  const [adding, setAdding] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const libraryAction = useCallback(async () => {
    setAdding(true);
    if (libraryAgent) {
      toast({
        description: "Redirecting to your library...",
        duration: 2000,
      });
      // Redirect to the library agent page
      router.push(`/library/agents/${libraryAgent.id}`);
      return;
    }
    try {
      const newLibraryAgent = await api.addMarketplaceAgentToLibrary(
        storeListingVersionId,
      );
      completeStep("MARKETPLACE_ADD_AGENT");
      router.push(`/library/agents/${newLibraryAgent.id}`);
      toast({
        title: "Agent Added",
        description: "Redirecting to your library...",
        duration: 2000,
      });
    } catch (error) {
      console.error("Failed to add agent to library:", error);
      toast({
        title: "Error",
        description: "Failed to add agent to library. Please try again.",
        variant: "destructive",
      });
    }
  }, [toast, api, storeListingVersionId, completeStep, router, libraryAgent]);

  const handleDownload = useCallback(async () => {
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
        toast({
          title: "Error",
          description: "Failed to download agent. Please try again.",
          variant: "destructive",
        });
      }
    };
    await downloadAgent();
    setDownloading(false);
  }, [setDownloading, api, storeListingVersionId, toast]);

  return (
    <div className="w-full max-w-[27rem] space-y-7">
      {/* Top part */}
      <div className="space-y-[3.25rem]">
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
        {/* TODO - Add one more button */}
        <div className="flex w-full items-center gap-3">
          {user && (
            <AutogptButton onClick={libraryAction} disabled={adding} icon>
              {libraryAgent ? "Run agent" : "Add to library"}
            </AutogptButton>
          )}
          <AutogptButton
            variant={"secondary"}
            onClick={handleDownload}
            disabled={downloading}
          >
            Download agent
          </AutogptButton>
        </div>

        {/* Runs and ratings */}
        <div className="flex w-full items-center gap-10">
          <div className="flex items-center gap-1.5">
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
              <Chip
                key={index}
                className="hover:border-zinc-400 hover:bg-white"
              >
                {category}
              </Chip>
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
