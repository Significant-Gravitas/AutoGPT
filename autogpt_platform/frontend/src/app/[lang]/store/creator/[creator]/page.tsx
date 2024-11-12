import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { CreatorDetails } from "@/components/agptui/composite/CreatorDetails";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";


export async function generateStaticParams() {
  const api = new AutoGPTServerAPI();
  const creators = await api.getStoreCreators({ featured: true });
  return creators.creators.map((creator) => ({
    creator: creator.username,
  }));
}

export default async function Page({
  params,
}: {
  params: { lang: string; creator: string };
}) {
  const { creator } = params;
  const api = new AutoGPTServerAPI();
  const creatorDetails = await api.getStoreCreator(creator);

  return (
    <div className="flex w-full flex-col items-center justify-center px-4">
      <CreatorDetails
        name={creatorDetails.name}
        username={creatorDetails.username}
        description={creatorDetails.description}
        avgRating={creatorDetails.agent_rating}
        agentCount={creatorDetails.agent_runs}
        topCategories={creatorDetails.top_categories}
        otherLinks={creatorDetails.links}
        avatarSrc={creatorDetails.avatar_url}
      />
    </div>
  );
}