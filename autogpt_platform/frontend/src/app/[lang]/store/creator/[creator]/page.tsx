import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import {
  CreatorDetails as Creator,
  StoreAgent,
} from "@/lib/autogpt-server-api";
import { CreatorDetails } from "@/components/agptui/composite/CreatorDetails";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { BreadCrumbs } from "@/components/agptui/BreadCrumbs";
import { Metadata } from "next";

export async function generateMetadata({
  params,
}: {
  params: { creator: string };
}): Promise<Metadata> {
  const api = new AutoGPTServerAPI();
  const creator = await api.getStoreCreator(params.creator);

  return {
    title: `${creator.name} - AutoGPT Store`,
    description: creator.description,
  };
}

export async function generateStaticParams() {
  const api = new AutoGPTServerAPI();
  const creators = await api.getStoreCreators({ featured: true });
  return creators.creators.map((creator) => ({
    creator: creator.username,
    lang: "en",
  }));
}

export default async function Page({
  params,
}: {
  params: { lang: string; creator: string };
}) {
  const api = new AutoGPTServerAPI();
  const creator = await api.getStoreCreator(params.creator);
  const creatorAgents = await api.getStoreAgents({ creator: params.creator });
  const agents = creatorAgents.agents;

  return (
    <>
      <div className="flex w-full flex-col items-center justify-center px-4">
        <div className="mt-8">
          <BreadCrumbs
            items={[
              { name: "Store", link: "/store" },
              { name: creator.name, link: "#" },
            ]}
          />
          <CreatorDetails
            avatarSrc={creator.avatar_url}
            name={creator.name}
            username={creator.username}
            description={creator.description}
            avgRating={creator.agent_rating}
            agentCount={creator.agent_runs}
            topCategories={creator.top_categories}
            otherLinks={creator.links}
          />
        </div>
        <div className="mt-16">
          <AgentsSection
            agents={agents}
            hideAvatars={true}
            sectionTitle={`Agents by ${creator.name}`}
          />
        </div>
      </div>
    </>
  );
}
