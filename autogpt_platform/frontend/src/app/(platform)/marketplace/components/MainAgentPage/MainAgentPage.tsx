"use client";
import { okData } from "@/app/api/helpers";
import { Separator } from "@/components/__legacy__/ui/separator";
import { Button } from "@/components/atoms/Button/Button";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ArrowLeftIcon } from "@phosphor-icons/react";
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
    return <AgentPageLoading />;
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
      <main className="mt-5 px-4 pb-12">
        <div className="mb-4 flex items-center justify-between px-4 md:!-mb-3">
          <Button
            variant="ghost"
            size="small"
            as="NextLink"
            href="/marketplace"
            className="relative -left-2 lg:!-left-4"
            leftIcon={<ArrowLeftIcon size={16} />}
          >
            Go back
          </Button>
          <div className="hidden md:block">
            <Breadcrumbs items={breadcrumbs} />
          </div>
        </div>

        <div className="mt-0 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 lg:mt-8 lg:flex-row lg:gap-12">
          <div className="w-full lg:w-2/5">
            <AgentInfo
              user={user}
              agentId={agentData.active_version_id ?? "–"}
              name={agentData.agent_name ?? ""}
              creator={agentData.creator ?? ""}
              creatorAvatar={agentData.creator_avatar ?? ""}
              shortDescription={agentData.sub_heading ?? ""}
              longDescription={agentData.description ?? ""}
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
        <Separator className="my-6 bg-transparent" />
        {otherAgents && (
          <AgentsSection
            agents={otherAgents.agents}
            sectionTitle={`Other agents by ${agentData.creator ?? ""}`}
          />
        )}
        <Separator className="mb-[25px] mt-[60px] bg-transparent" />
        {similarAgents && similarAgents.agents.length > 0 ? (
          <AgentsSection
            agents={similarAgents.agents}
            sectionTitle="Similar agents"
          />
        ) : null}
        <BecomeACreator />
      </main>
    </div>
  );
}
