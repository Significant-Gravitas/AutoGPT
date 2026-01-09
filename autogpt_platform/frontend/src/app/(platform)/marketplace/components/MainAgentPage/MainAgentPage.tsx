"use client";
import { Separator } from "@/components/__legacy__/ui/separator";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { okData } from "@/app/api/helpers";
import { MarketplaceAgentPageParams } from "../../agent/[creator]/[slug]/page";
import { AgentImages } from "../AgentImages/AgentImage";
import { AgentInfo } from "../AgentInfo/AgentInfo";
import { AgentPageLoading } from "../AgentPageLoading";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import { useMainAgentPage } from "./useMainAgentPage";

interface Props {
  params: MarketplaceAgentPageParams;
}

export function MainAgentPage({ params }: Props) {
  const {
    agent,
    user,
    isLoading,
    hasError,
    similarAgents,
    otherAgents,
    libraryAgent,
  } = useMainAgentPage({ params });

  if (isLoading) {
    return (
      <div className="mx-auto w-full max-w-[1360px]">
        <main className="px-4">
          <div className="flex h-[600px] items-center justify-center">
            <AgentPageLoading />
          </div>
        </main>
      </div>
    );
  }

  if (hasError) {
    return (
      <div className="mx-auto w-full max-w-[1360px]">
        <main className="px-4">
          <div className="flex min-h-[400px] items-center justify-center">
            <ErrorCard
              isSuccess={false}
              responseError={{ message: "Failed to load agent data" }}
              context="agent page"
              onRetry={() => window.location.reload()}
              className="w-full max-w-md"
            />
          </div>
        </main>
      </div>
    );
  }

  const agentData = okData(agent);
  if (!agentData) {
    return (
      <div className="mx-auto w-full max-w-[1360px]">
        <main className="px-4">
          <div className="flex min-h-[400px] items-center justify-center">
            <ErrorCard
              isSuccess={false}
              responseError={{ message: "Agent not found" }}
              context="agent page"
            />
          </div>
        </main>
      </div>
    );
  }

  const breadcrumbs = [
    { name: "Marketplace", link: "/marketplace" },
    {
      name: agentData.creator ?? "",
      link: `/marketplace/creator/${encodeURIComponent(agentData.creator ?? "")}`,
    },
    { name: agentData.agent_name ?? "", link: "#" },
  ];

  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="mt-5 px-4">
        <Breadcrumbs items={breadcrumbs} />

        <div className="mt-4 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 md:mt-8 md:flex-row md:gap-8">
          <div className="w-full md:w-auto md:shrink-0">
            <AgentInfo
              user={user}
              agentId={agentData.active_version_id ?? "–"}
              name={agentData.agent_name ?? ""}
              creator={agentData.creator ?? ""}
              shortDescription={agentData.sub_heading ?? ""}
              longDescription={agentData.description ?? ""}
              rating={agentData.rating ?? 0}
              runs={agentData.runs ?? 0}
              categories={agentData.categories ?? []}
              lastUpdated={
                agentData.last_updated?.toISOString() ??
                new Date().toISOString()
              }
              version={
                agentData.versions
                  ? Math.max(
                      ...agentData.versions.map((v: string) => parseInt(v, 10)),
                    ).toString()
                  : "1"
              }
              storeListingVersionId={agentData.store_listing_version_id ?? ""}
              isAgentAddedToLibrary={Boolean(libraryAgent)}
              creatorSlug={params.creator}
              agentSlug={params.slug}
            />
          </div>
          <AgentImages
            images={(() => {
              const orderedImages: string[] = [];

              // 1. YouTube/Overview video (if it exists)
              if (agentData.agent_video) {
                orderedImages.push(agentData.agent_video);
              }

              // 2. First image (hero)
              if (agentData.agent_image?.length > 0) {
                orderedImages.push(agentData.agent_image[0]);
              }

              // 3. Agent Output Demo (if it exists)
              if (agentData.agent_output_demo) {
                orderedImages.push(agentData.agent_output_demo);
              }

              // 4. Additional images
              if (agentData.agent_image && agentData.agent_image.length > 1) {
                orderedImages.push(...agentData.agent_image.slice(1));
              }

              return orderedImages;
            })()}
          />
        </div>
        <Separator className="mb-[25px] mt-[60px]" />
        {otherAgents && (
          <AgentsSection
            margin="32px"
            agents={otherAgents.agents}
            sectionTitle={`Other agents by ${agentData.creator ?? ""}`}
          />
        )}
        <Separator className="mb-[25px] mt-[60px]" />
        {similarAgents && (
          <AgentsSection
            margin="32px"
            agents={similarAgents.agents}
            sectionTitle="Similar agents"
          />
        )}
        <BecomeACreator />
      </main>
    </div>
  );
}
