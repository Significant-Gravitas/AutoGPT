import BackendAPI from "@/lib/autogpt-server-api";
import { BreadCrumbs } from "@/components/agptui/BreadCrumbs";
import { AgentInfo } from "@/components/agptui/AgentInfo";
import { AgentImages } from "@/components/agptui/AgentImages";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { BecomeACreator } from "@/components/agptui/BecomeACreator";
import { Separator } from "@/components/ui/separator";
import { Metadata } from "next";

export async function generateMetadata({
  params,
}: {
  params: { creator: string; slug: string };
}): Promise<Metadata> {
  const api = new BackendAPI();
  const agent = await api.getStoreAgent(params.creator, params.slug);

  return {
    title: `${agent.agent_name} - AutoGPT Store`,
    description: agent.description,
  };
}

// export async function generateStaticParams() {
//   const api = new BackendAPI();
//   const agents = await api.getStoreAgents({ featured: true });
//   return agents.agents.map((agent) => ({
//     creator: agent.creator,
//     slug: agent.slug,
//   }));
// }

export default async function Page({
  params,
}: {
  params: { creator: string; slug: string };
}) {
  const creator_lower = params.creator.toLowerCase();
  const api = new BackendAPI();
  const agent = await api.getStoreAgent(creator_lower, params.slug);
  const otherAgents = await api.getStoreAgents({ creator: creator_lower });
  const similarAgents = await api.getStoreAgents({
    // We are using slug as we know its has been sanitized and is not null
    search_query: agent.slug.replace(/-/g, " "),
  });

  const breadcrumbs = [
    { name: "Store", link: "/marketplace" },
    {
      name: agent.creator,
      link: `/marketplace/creator/${encodeURIComponent(agent.creator)}`,
    },
    { name: agent.agent_name, link: "#" },
  ];

  return (
    <div className="mx-auto w-screen max-w-[1360px]">
      <main className="mt-5 px-4">
        <BreadCrumbs items={breadcrumbs} />

        <div className="mt-4 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 md:mt-8 md:flex-row md:gap-8">
          <div className="w-full md:w-auto md:shrink-0">
            <AgentInfo
              name={agent.agent_name}
              creator={agent.creator}
              shortDescription={agent.description}
              longDescription={agent.description}
              rating={agent.rating}
              runs={agent.runs}
              categories={agent.categories}
              lastUpdated={agent.updated_at}
              version={agent.versions[agent.versions.length - 1]}
              storeListingVersionId={agent.store_listing_version_id}
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
        <Separator className="mb-[25px] mt-6" />
        <AgentsSection
          agents={otherAgents.agents}
          sectionTitle={`Other agents by ${agent.creator}`}
        />
        <Separator className="mb-[25px] mt-6" />
        <AgentsSection
          agents={similarAgents.agents}
          sectionTitle="Similar agents"
        />
        <BecomeACreator
          title="Become a Creator"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
}
