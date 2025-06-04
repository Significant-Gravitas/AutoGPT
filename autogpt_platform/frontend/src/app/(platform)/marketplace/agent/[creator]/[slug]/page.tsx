import BackendAPI from "@/lib/autogpt-server-api";
import { BreadCrumbs } from "@/components/agptui/BreadCrumbs";
import { AgentInfo } from "@/components/agptui/AgentInfo";
import { AgentImages } from "@/components/agptui/AgentImages";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { BecomeACreator } from "@/components/agptui/BecomeACreator";
import { Separator } from "@/components/ui/separator";
import { Metadata } from "next";
import getServerUser from "@/lib/supabase/getServerUser";

// Force dynamic rendering to avoid static generation issues with cookies
export const dynamic = "force-dynamic";

type MarketplaceAgentPageParams = { creator: string; slug: string };

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<MarketplaceAgentPageParams>;
}): Promise<Metadata> {
  const api = new BackendAPI();
  const params = await _params;
  const agent = await api.getStoreAgent(params.creator, params.slug);

  return {
    title: `${agent.agent_name} - AutoGPT Marketplace`,
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

export default async function MarketplaceAgentPage({
  params: _params,
}: {
  params: Promise<MarketplaceAgentPageParams>;
}) {
  const params = await _params;
  const creator_lower = params.creator.toLowerCase();
  const { user } = await getServerUser();
  const api = new BackendAPI();
  const agent = await api.getStoreAgent(creator_lower, params.slug);
  const otherAgents = await api.getStoreAgents({ creator: creator_lower });
  const similarAgents = await api.getStoreAgents({
    // We are using slug as we know its has been sanitized and is not null
    search_query: agent.slug.replace(/-/g, " "),
  });
  const libraryAgent = user
    ? await api
        .getLibraryAgentByStoreListingVersionID(agent.active_version_id || "")
        .catch((error) => {
          console.error("Failed to fetch library agent:", error);
          return null;
        })
    : null;

  const breadcrumbs = [
    { name: "Marketplace", link: "/marketplace" },
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
              user={user}
              name={agent.agent_name}
              creator={agent.creator}
              shortDescription={agent.sub_heading}
              longDescription={agent.description}
              rating={agent.rating}
              runs={agent.runs}
              categories={agent.categories}
              lastUpdated={agent.updated_at}
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
        <Separator className="mb-[25px] mt-[60px]" />
        <AgentsSection
          margin="32px"
          agents={otherAgents.agents}
          sectionTitle={`Other agents by ${agent.creator}`}
        />
        <Separator className="mb-[25px] mt-[60px]" />
        <AgentsSection
          margin="32px"
          agents={similarAgents.agents}
          sectionTitle="Similar agents"
        />
        <Separator className="mb-[25px] mt-[60px]" />
        <BecomeACreator
          title="Become a Creator"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
}
