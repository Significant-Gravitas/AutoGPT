import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import {
  CreatorDetails as Creator,
  StoreAgent,
} from "@/lib/autogpt-server-api";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { BreadCrumbs } from "@/components/agptui/BreadCrumbs";
import { Metadata } from "next";
import { CreatorInfoCard } from "@/components/agptui/CreatorInfoCard";
import { CreatorLinks } from "@/components/agptui/CreatorLinks";

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

// export async function generateStaticParams() {
//   const api = new AutoGPTServerAPI();
//   const creators = await api.getStoreCreators({ featured: true });
//   return creators.creators.map((creator) => ({
//     creator: creator.username,
//   }));
// }

export default async function Page({
  params,
}: {
  params: { creator: string };
}) {
  const api = new AutoGPTServerAPI();

  try {
    const creator = await api.getStoreCreator(params.creator);
    const creatorAgents = await api.getStoreAgents({ creator: params.creator });

    return (
      <div className="mx-auto w-screen max-w-[1360px]">
        <main className="px-4 md:mt-4 lg:mt-8">
          <BreadCrumbs
            items={[
              { name: "Store", link: "/store" },
              { name: creator.name, link: "#" },
            ]}
          />

          <div className="mt-4 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 md:mt-8 md:flex-row md:gap-8">
            <div className="w-full md:w-auto md:shrink-0">
              <CreatorInfoCard
                username={creator.name}
                handle={creator.username}
                avatarSrc={creator.avatar_url}
                categories={creator.top_categories}
                averageRating={creator.agent_rating}
                totalRuns={creator.agent_runs}
              />
            </div>
            <div className="flex min-w-0 flex-1 flex-col gap-4 sm:gap-6 md:gap-8">
              <div className="font-neue text-2xl font-normal leading-normal text-neutral-900 sm:text-3xl md:text-[35px] md:leading-[45px]">
                {creator.description}
              </div>
              <CreatorLinks links={creator.links} />
            </div>
          </div>
          <div className="mt-8 sm:mt-12 md:mt-16">
            <hr className="w-full bg-neutral-700" />
            <AgentsSection
              agents={creatorAgents.agents}
              hideAvatars={true}
              sectionTitle={`Agents by ${creator.name}`}
            />
          </div>
        </main>
      </div>
    );
  } catch (error) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="font-neue text-2xl text-neutral-900">
          Creator not found
        </div>
      </div>
    );
  }
}
