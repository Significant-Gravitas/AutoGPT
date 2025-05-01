import BackendAPI from "@/lib/autogpt-server-api";
import {
  CreatorDetails as Creator,
  StoreAgent,
} from "@/lib/autogpt-server-api";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { BreadCrumbs } from "@/components/agptui/BreadCrumbs";
import { Metadata } from "next";
import { CreatorInfoCard } from "@/components/agptui/CreatorInfoCard";
import { CreatorLinks } from "@/components/agptui/CreatorLinks";
import { Separator } from "@/components/ui/separator";

export async function generateMetadata({
  params,
}: {
  params: { creator: string };
}): Promise<Metadata> {
  const api = new BackendAPI();
  const creator = await api.getStoreCreator(params.creator.toLowerCase());

  return {
    title: `${creator.name} - AutoGPT Store`,
    description: creator.description,
  };
}

// export async function generateStaticParams() {
//   const api = new BackendAPI();
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
  const api = new BackendAPI();

  try {
    const creator = await api.getStoreCreator(params.creator);
    const creatorAgents = await api.getStoreAgents({ creator: params.creator });

    return (
      <main className="px-10">
        <BreadCrumbs
          items={[
            { name: "Store", link: "/marketplace" },
            { name: creator.name, link: "#" },
          ]}
        />

        <div className="mt-4 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 md:mt-8 md:flex-row md:gap-8">
          <div>
            <CreatorInfoCard
              username={creator.name}
              handle={creator.username}
              avatarSrc={creator.avatar_url}
              categories={creator.top_categories}
              averageRating={creator.agent_rating}
              totalRuns={creator.agent_runs}
            />
          </div>
          <div className="flex-1 space-y-7">
            <div>
              <p className="font-sans text-base font-medium text-zinc-800">
                About
              </p>
              <h1 className="font-poppins text-4xl font-normal leading-[3.25rem] text-zinc-800">
                {creator.description}
              </h1>
            </div>

            <CreatorLinks links={creator.links} />
          </div>
        </div>

        <div className="mt-8 sm:mt-12 md:mt-16 lg:pb-[58px]">
          <Separator className="mb-6 bg-gray-200" />
          <AgentsSection
            agents={creatorAgents.agents}
            hideAvatars={true}
            sectionTitle={`Agents by ${creator.name}`}
          />
        </div>
      </main>
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
