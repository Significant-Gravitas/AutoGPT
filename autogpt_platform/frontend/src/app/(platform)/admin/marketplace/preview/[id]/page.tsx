"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft } from "@phosphor-icons/react";
import { AgentInfo } from "@/app/(platform)/marketplace/components/AgentInfo/AgentInfo";
import { AgentImages } from "@/app/(platform)/marketplace/components/AgentImages/AgentImage";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { previewAsAdmin, addToLibraryAsAdmin } from "../../actions";
import { useToast } from "@/components/molecules/Toast/use-toast";

export default function AdminPreviewPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const { toast } = useToast();
  const [data, setData] = useState<StoreAgentDetails | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isAddingToLibrary, setIsAddingToLibrary] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const result = await previewAsAdmin(params.id);
        setData(result as StoreAgentDetails);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load preview");
      } finally {
        setIsLoading(false);
      }
    }
    load();
  }, [params.id]);

  async function handleAddToLibrary() {
    setIsAddingToLibrary(true);
    try {
      await addToLibraryAsAdmin(params.id);
      toast({
        title: "Added to Library",
        description: "Agent has been added to your library for review.",
        duration: 3000,
      });
    } catch (e) {
      toast({
        title: "Error",
        description:
          e instanceof Error ? e.message : "Failed to add agent to library.",
        variant: "destructive",
      });
    } finally {
      setIsAddingToLibrary(false);
    }
  }

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-muted-foreground">Loading preview...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4">
        <p className="text-destructive">{error || "Preview not found"}</p>
        <button
          onClick={() => router.back()}
          className="text-muted-foreground underline"
        >
          Go back
        </button>
      </div>
    );
  }

  const allMedia = [
    ...(data.agent_video ? [data.agent_video] : []),
    ...(data.agent_output_demo ? [data.agent_output_demo] : []),
    ...data.agent_image,
  ];

  return (
    <div className="container mx-auto max-w-7xl px-4 py-6">
      <div className="mb-6 flex items-center justify-between">
        <button
          onClick={() => router.back()}
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft size={16} />
          Back to Admin Marketplace
        </button>

        <div className="flex items-center gap-3">
          <span className="rounded-md bg-amber-500/20 px-3 py-1 text-sm font-medium text-amber-600">
            Admin Preview
            {!data.has_approved_version && " — Pending Approval"}
          </span>
          <button
            onClick={handleAddToLibrary}
            disabled={isAddingToLibrary}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {isAddingToLibrary ? "Adding..." : "Add to My Library"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <AgentInfo
            user={null}
            agentId={data.graph_id}
            name={data.agent_name}
            creator={data.creator}
            creatorAvatar={data.creator_avatar}
            shortDescription={data.sub_heading}
            longDescription={data.description}
            runs={data.runs}
            categories={data.categories}
            lastUpdated={String(data.last_updated)}
            version={data.versions[0] || "1"}
            storeListingVersionId={data.store_listing_version_id}
            isAgentAddedToLibrary={false}
          />
        </div>
        <div className="lg:col-span-3">
          {allMedia.length > 0 ? (
            <AgentImages images={allMedia} />
          ) : (
            <div className="flex h-64 items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25">
              <p className="text-muted-foreground">
                No images or videos submitted
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Fields not shown in AgentInfo but important for admin review */}
      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
        {data.instructions && (
          <div className="rounded-lg border p-4">
            <h3 className="mb-2 text-sm font-semibold text-muted-foreground">
              Instructions
            </h3>
            <p className="whitespace-pre-wrap text-sm">{data.instructions}</p>
          </div>
        )}
        {data.recommended_schedule_cron && (
          <div className="rounded-lg border p-4">
            <h3 className="mb-2 text-sm font-semibold text-muted-foreground">
              Recommended Schedule
            </h3>
            <code className="rounded bg-muted px-2 py-1 text-sm">
              {data.recommended_schedule_cron}
            </code>
          </div>
        )}
        <div className="rounded-lg border p-4">
          <h3 className="mb-2 text-sm font-semibold text-muted-foreground">
            Slug
          </h3>
          <code className="rounded bg-muted px-2 py-1 text-sm">
            {data.slug}
          </code>
        </div>
      </div>
    </div>
  );
}
