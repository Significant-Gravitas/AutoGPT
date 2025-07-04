"use client";
import { Separator } from "@/components/ui/separator";
import { AgentImages } from "../AgentImages/AgentImages";
import { AgentInfo } from "../AgentInfo/AgentInfo";
import { BreadCrumbs } from "../BreadCrumbs/BreadCrumbs";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import { getBreadcrumbs } from "./helper";
import { useMainAgentPage } from "./useMainAgentPage";
import { MarketplaceAgentPageParams } from "../../agent/[creator]/[slug]/page";

interface MainAgentPageProps {
  params: MarketplaceAgentPageParams;
}

export const MainAgentPage = ({ params }: MainAgentPageProps) => {
  const {
    libraryAgent,
    similarAgents,
    otherAgents,
    agentData: agent,
    user,
  } = useMainAgentPage({ params });

  return (
    <div className="mx-auto w-screen max-w-[1360px]">
      <main className="mt-5 px-4">
        {agent && <BreadCrumbs items={getBreadcrumbs(agent)} />}

        {agent && (
          <div className="mt-4 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 md:mt-8 md:flex-row md:gap-8">
            <div className="w-full md:w-auto md:shrink-0">
              <AgentInfo
                user={user}
                name={agent.agent_name}
                creator={agent.creator}
                shortDescription={agent.sub_heading}
                longDescription={agent.description}
                rating={agent.rating}
                runs={agent.runs}
                categories={agent.categories}
                lastUpdated={""} // Not exist on backend
                version={agent.versions[agent.versions.length - 1]}
                storeListingVersionId={agent.store_listing_version_id}
                libraryAgent={libraryAgent}
              />
            </div>
            <AgentImages
              images={
                agent.agent_video
                  ? [agent.agent_video, ...agent.agent_image]
                  : agent.agent_image
              }
            />
          </div>
        )}
        <Separator className="mb-[25px] mt-[60px]" />
        {otherAgents && agent && (
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
