"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { AgentInfo } from "@/app/(platform)/marketplace/components/AgentInfo/AgentInfo";
import { AgentImages } from "@/app/(platform)/marketplace/components/AgentImages/AgentImage";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { previewAsAdmin, addToLibraryAsAdmin } from "../../actions";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

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
        <Text variant="large" tone="muted">
          Loading preview...
        </Text>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4">
        <Text variant="large" unmask={false} tone="danger">
          {error || "Preview not found"}
        </Text>
        <Button variant="link" onClick={() => router.back()}>
          Go back
        </Button>
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
        <Button
          variant="ghost"
          size="md"
          onClick={() => router.back()}
          leadingIcon={ArrowLeft02Icon}
        >
          Back to Admin Marketplace
        </Button>

        <div className="flex items-center gap-3">
          <span className="rounded-md bg-yellow-500/20 px-3 py-1 text-sm font-medium text-yellow-600">
            Admin Preview
            {!data.has_approved_version && " — Pending Approval"}
          </span>
          <Button
            variant="primary"
            size="md"
            onClick={handleAddToLibrary}
            disabled={isAddingToLibrary}
          >
            {isAddingToLibrary ? "Adding..." : "Add to My Library"}
          </Button>
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
              <Text variant="large" tone="muted">
                No images or videos submitted
              </Text>
            </div>
          )}
        </div>
      </div>

      {/* Fields not shown in AgentInfo but important for admin review */}
      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
        {data.instructions && (
          <div className="rounded-lg border p-4">
            <Text variant="body-medium" as="h3" tone="muted" className="mb-2">
              Instructions
            </Text>
            <Text variant="body" unmask={false} className="whitespace-pre-wrap">
              {data.instructions}
            </Text>
          </div>
        )}
        {data.recommended_schedule_cron && (
          <div className="rounded-lg border p-4">
            <Text variant="body-medium" as="h3" tone="muted" className="mb-2">
              Recommended Schedule
            </Text>
            <code className="rounded-sm bg-muted px-2 py-1 text-sm">
              {data.recommended_schedule_cron}
            </code>
          </div>
        )}
        <div className="rounded-lg border p-4">
          <Text variant="body-medium" as="h3" tone="muted" className="mb-2">
            Slug
          </Text>
          <code className="rounded-sm bg-muted px-2 py-1 text-sm">
            {data.slug}
          </code>
        </div>
      </div>
    </div>
  );
}
