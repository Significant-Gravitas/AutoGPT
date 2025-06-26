import BackendAPI from "@/lib/autogpt-server-api";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { BreadCrumbs } from "@/components/agptui/BreadCrumbs";
import { Metadata } from "next";
import { CreatorInfoCard } from "@/components/agptui/CreatorInfoCard";
import { CreatorLinks } from "@/components/agptui/CreatorLinks";
import { Separator } from "@/components/ui/separator";

// Force dynamic rendering to avoid static generation issues with cookies
export const dynamic = "force-dynamic";

type MarketplaceCreatorPageParams = { creator: string };

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<MarketplaceCreatorPageParams>;
}): Promise<Metadata> {
  const api = new BackendAPI();
  const params = await _params;
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
  params: _params,
}: {
  params: Promise<MarketplaceCreatorPageParams>;
}) {
  const api = new BackendAPI();
  const params = await _params;

  try {
    const creator = await api.getStoreCreator(params.creator);
    const creatorAgents = await api.getStoreAgents({ creator: params.creator });

    return (
      <div className="mx-auto w-screen max-w-[1360px]">
        <main className="mt-5 px-4">
          <BreadCrumbs
            items={[
              { name: "Store", link: "/marketplace" },
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
              <p className="text-underline-position-from-font text-decoration-skip-none text-left font-poppins text-base font-medium leading-6">
                About
              </p>
              <div
                className="text-[48px] font-normal leading-[59px] text-neutral-900 dark:text-zinc-50"
                style={{ whiteSpace: "pre-line" }}
              >
                {creator.description}
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
      </div>
    );
  } catch {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="text-2xl text-neutral-900">Creator not found</div>
      </div>
    );
  }
}
