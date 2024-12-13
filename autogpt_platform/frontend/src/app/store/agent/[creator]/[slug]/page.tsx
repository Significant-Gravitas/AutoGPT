import AutoGPTServerAPI from "@/lib/autogpt-server-api";
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
  const api = new AutoGPTServerAPI();
  const agent = await api.getStoreAgent(params.creator, params.slug);

  return {
    title: `${agent.agent_name} - AutoGPT Store`,
    description: agent.description,
  };
}

// export async function generateStaticParams() {
//   const api = new AutoGPTServerAPI();
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
  const api = new AutoGPTServerAPI();
  const agent = await api.getStoreAgent(params.creator, params.slug);
  const otherAgents = await api.getStoreAgents({ creator: params.creator });
  const similarAgents = await api.getStoreAgents({
    search_query: agent.categories[0],
  });

  const breadcrumbs = [
    { name: "Store", link: "/store" },
    { name: agent.creator, link: `/store/creator/${agent.creator}` },
    { name: agent.agent_name, link: "#" },
  ];

  return (
    <div className="mx-auto w-screen max-w-[1360px]">
      <main className="px-4 md:mt-4 lg:mt-8">
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
            />
          </div>
          <AgentImages images={agent.agent_image} />
        </div>
        <Separator className="my-6" />
        <AgentsSection
          agents={otherAgents.agents}
          sectionTitle={`Other agents by ${agent.creator}`}
        />
        <Separator className="my-6" />
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
