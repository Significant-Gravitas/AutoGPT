"use client";
import { Separator } from "@/components/__legacy__/ui/separator";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { MarketplaceAgentPageParams } from "../../agent/[creator]/[slug]/page";
import { AgentImages } from "../AgentImages/AgentImage";
import { AgentInfo } from "../AgentInfo/AgentInfo";
import { AgentPageLoading } from "../AgentPageLoading";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import { useMainAgentPage } from "./useMainAgentPage";

type MainAgentPageProps = {
  params: MarketplaceAgentPageParams;
};

export const MainAgentPage = ({ params }: MainAgentPageProps) => {
  const {
    agent,
    otherAgents,
    similarAgents,
    libraryAgent,
    isLoading,
    hasError,
    user,
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

  if (!agent) {
    return (
      <div className="mx-auto w-full max-w-[1360px]">
        <main className="px-4">
          <div className="flex min-h-[400px] items-center justify-center">
            <ErrorCard
              isSuccess={false}
              responseError={{ message: "Agent not found" }}
              context="agent page"
              onRetry={() => window.location.reload()}
              className="w-full max-w-md"
            />
          </div>
        </main>
      </div>
    );
  }

  const breadcrumbs = [
    { name: "Marketplace", link: "/marketplace" },
    {
      name: agent.creator,
      link: `/marketplace/creator/${encodeURIComponent(agent.creator)}`,
    },
    { name: agent.agent_name, link: "#" },
  ];

  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="mt-5 px-4">
        <Breadcrumbs items={breadcrumbs} />

        <div className="mt-4 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 md:mt-8 md:flex-row md:gap-8">
          <div className="w-full md:w-auto md:shrink-0">
            <AgentInfo
              user={user}
              agentId={agent.active_version_id ?? "–"}
              name={agent.agent_name}
              creator={agent.creator}
              shortDescription={agent.sub_heading}
              longDescription={agent.description}
              rating={agent.rating}
              runs={agent.runs}
              categories={agent.categories}
              lastUpdated={agent.last_updated.toISOString()}
              version={agent.versions[agent.versions.length - 1]}
              storeListingVersionId={agent.store_listing_version_id}
              isAgentAddedToLibrary={Boolean(libraryAgent)}
            />
          </div>
          <AgentImages
            images={(() => {
              const orderedImages: string[] = [];

              // 1. YouTube/Overview video (if it exists)
              if (agent.agent_video) {
                orderedImages.push(agent.agent_video);
              }

              // 2. First image (hero)
              if (agent.agent_image.length > 0) {
                orderedImages.push(agent.agent_image[0]);
              }

              // 3. Agent Output Demo (if it exists)
              if ((agent as any).agent_output_demo) {
                orderedImages.push((agent as any).agent_output_demo);
              }

              // 4. Additional images
              if (agent.agent_image.length > 1) {
                orderedImages.push(...agent.agent_image.slice(1));
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
            sectionTitle={`Other agents by ${agent.creator}`}
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
        <Separator className="mb-[25px] mt-[60px]" />
        <BecomeACreator
          title="Become a Creator"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
};
