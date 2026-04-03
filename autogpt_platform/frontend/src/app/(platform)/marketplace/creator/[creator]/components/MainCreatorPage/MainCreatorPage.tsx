"use client";

import { AgentsSection } from "@/app/(platform)/marketplace/components/AgentsSection/AgentsSection";
import { CreatorLinks } from "@/app/(platform)/marketplace/components/CreatorLinks/CreatorLinks";
import { CreatorPageLoading } from "@/app/(platform)/marketplace/components/CreatorPageLoading";
import { MarketplaceCreatorPageParams } from "@/app/(platform)/marketplace/creator/[creator]/page";

import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ArrowLeftIcon } from "@phosphor-icons/react";
import { useMainCreatorPage } from "./useMainCreatorPage";

interface Props {
  params: MarketplaceCreatorPageParams;
}

export function MainCreatorPage({ params }: Props) {
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

  if (!creator) return null;

  const breadcrumbs = [
    { name: "Marketplace", link: "/marketplace" },
    { name: creator.name, link: "#" },
  ];

  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="mt-5 px-4 pb-12">
        <div className="mb-4 flex items-center justify-between px-4 md:!-mb-3">
          <Button
            variant="ghost"
            size="small"
            as="NextLink"
            href="/marketplace"
            className="relative -left-2 lg:!-left-4"
            leftIcon={<ArrowLeftIcon size={16} />}
          >
            Go back
          </Button>
          <div className="hidden md:block">
            <Breadcrumbs items={breadcrumbs} />
          </div>
        </div>

        <div className="mt-0 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 lg:mt-8 lg:flex-row lg:gap-12">
          {/* Creator info - left side */}
          <div className="w-full lg:w-2/5">
            <div className="w-full px-4 sm:px-6 lg:px-0">
              <div className="max-w-[45rem] rounded-2xl bg-gradient-to-r from-blue-200 to-indigo-200 p-[1px]">
                <div className="flex flex-col rounded-[calc(1rem-2px)] bg-gray-50 p-4">
                  {/* Avatar */}
                  <Avatar className="mb-4 h-20 w-20 sm:h-24 sm:w-24">
                    {creator.avatar_url && (
                      <AvatarImage
                        src={creator.avatar_url}
                        alt={`${creator.name} avatar`}
                      />
                    )}
                    <AvatarFallback size={96}>
                      {creator.name.charAt(0)}
                    </AvatarFallback>
                  </Avatar>

                  {/* Name */}
                  <Text
                    variant="h2"
                    className="mb-1"
                    data-testid="creator-title"
                  >
                    {creator.name}
                  </Text>

                  {/* Handle */}
                  <Text variant="body" className="mb-4 text-neutral-500">
                    @{creator.username}
                  </Text>

                  {/* Description */}
                  <Text
                    variant="body"
                    className="mb-6 leading-relaxed text-neutral-600"
                    data-testid="creator-description"
                  >
                    {creator.description}
                  </Text>

                  {/* Categories */}
                  {creator.top_categories.length > 0 && (
                    <div className="mb-6">
                      <Text variant="h5" className="mb-2">
                        Top categories
                      </Text>
                      <div className="flex flex-wrap gap-2">
                        {creator.top_categories.map((category, index) => (
                          <Badge
                            variant="info"
                            key={index}
                            className="border border-purple-100 bg-purple-50 text-purple-800"
                          >
                            {category}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Links */}
                  <CreatorLinks links={creator.links} />
                </div>
              </div>
            </div>
          </div>

          {/* Right side - empty for now, keeps layout consistent */}
          <div className="hidden lg:block lg:w-3/5" />
        </div>

        <div className="my-18" />

        {creatorAgents && (
          <AgentsSection
            agents={creatorAgents.agents}
            hideAvatars
            sectionTitle={`Agents by ${creator.name}`}
          />
        )}
      </main>
    </div>
  );
}
