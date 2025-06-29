"use client";
import { Separator } from "@/components/ui/separator";
import { AgentImages } from "../AgentImages/AgentImages";
import { AgentInfo } from "../AgentInfo/AgentInfo";
import { BreadCrumbs } from "../BreadCrumbs/BreadCrumbs";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import { getBreadcrumbs } from "./helper";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import {
  useGetV2GetSpecificAgent,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import { MarketplaceAgentPageParams } from "../../agent/[creator]/[slug]/page";
import { useGetV2GetAgentByStoreId } from "@/app/api/__generated__/endpoints/library/library";
import { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

interface MainAgentPageProps {
  params: MarketplaceAgentPageParams;
}

export const MainAgentPage = ({ params }: MainAgentPageProps) => {
  const creator_lower = params.creator.toLowerCase();
  const { user } = useSupabase();
  const { data: agent } = useGetV2GetSpecificAgent(creator_lower, params.slug, {
    query: {
      select: (x) => {
        return x.data as StoreAgentDetails;
      },
      enabled: !!user,
    },
  });

  const { data: otherAgents } = useGetV2ListStoreAgents(
    {
      creator: creator_lower,
    },
    {
      query: {
        select: (x) => {
          return x.data as StoreAgentsResponse;
        },
      },
    },
  );

  const { data: similarAgents } = useGetV2ListStoreAgents(
    {
      search_query: agent?.slug.replace(/-/g, " "),
    },
    {
      query: {
        enabled: !!agent,
        select: (x) => {
          return x.data as StoreAgentsResponse;
        },
      },
    },
  );

  const { data: libraryAgent } = useGetV2GetAgentByStoreId(
    agent?.active_version_id || "",
    {
      query: {
        enabled: !!agent,
        select: (x) => {
          return x.data as LibraryAgent | null;
        },
      },
    },
  );

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
