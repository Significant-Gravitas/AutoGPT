"use client";

import { Separator } from "@/components/__legacy__/ui/separator";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { MarketplaceCreatorPageParams } from "../../creator/[creator]/page";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { CreatorInfoCard } from "../CreatorInfoCard/CreatorInfoCard";
import { CreatorLinks } from "../CreatorLinks/CreatorLinks";
import { useMainCreatorPage } from "./useMainCreatorPage";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { CreatorPageLoading } from "../CreatorPageLoading";

interface MainCreatorPageProps {
  params: MarketplaceCreatorPageParams;
}

export const MainCreatorPage = ({ params }: MainCreatorPageProps) => {
  const { creatorAgents, creator, isLoading, hasError } = useMainCreatorPage({
    params,
  });

  if (isLoading) return <CreatorPageLoading />;

  if (hasError) {
    return (
      <div className="mx-auto w-full max-w-[1360px]">
        <div className="flex min-h-[60vh] items-center justify-center">
          <ErrorCard
            isSuccess={false}
            responseError={{ message: "Failed to load creator data" }}
            context="creator page"
            onRetry={() => window.location.reload()}
            className="w-full max-w-md"
          />
        </div>
      </div>
    );
  }

  if (creator)
    return (
      <div className="mx-auto w-full max-w-[1360px]">
        <main className="mt-5 px-4">
          <Breadcrumbs
            items={[
              { name: "Marketplace", link: "/marketplace" },
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
                data-testid="creator-description"
              >
                {creator.description}
              </div>

              <CreatorLinks links={creator.links} />
            </div>
          </div>
          <div className="mt-8 sm:mt-12 md:mt-16 lg:pb-[58px]">
            <Separator className="mb-6 bg-gray-200" />
            {creatorAgents && (
              <AgentsSection
                agents={creatorAgents.agents}
                hideAvatars={true}
                sectionTitle={`Agents by ${creator.name}`}
              />
            )}
          </div>
        </main>
      </div>
    );
};
