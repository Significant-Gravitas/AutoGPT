"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { UserAiIcon } from "@hugeicons/core-free-icons";
import { useTrackFunnelViewOnce } from "@/services/experts/use-track-funnel-view-once";
import { SectionHeader } from "../SectionHeader";
import { ExpertCard } from "./components/ExpertCard";
import { useExpertsSection } from "./useExpertsSection";

const RAISE_LABEL = "Create an Expert";
const RAISE_HREF = "/raise";

interface Props {
  category?: string | null;
}

export function ExpertsSection({ category }: Props) {
  const { isLoggedIn, templates, hiredTemplateIds, isLoading, isError } =
    useExpertsSection({ category });

  useTrackFunnelViewOnce(
    "experts_section_viewed",
    !isLoading && !isError && templates.length > 0,
  );

  if (isError || (!isLoading && templates.length === 0)) {
    // Under a category filter an empty shelf means "no experts in this
    // category", so the whole section goes rather than inviting a raise.
    // Only on a successful empty response: a failed request is not an answer
    // about the category, and still deserves the fallback below.
    if (!isError && category) return null;
    // Raising an expert needs no roster templates, so the second door
    // stays open even when the template list is empty or failed to load.
    // It needs an account, though, so visitors get nothing here.
    if (!isLoggedIn) return null;
    return (
      <section id="experts" className="mb-20 scroll-mt-24">
        <Button
          as="NextLink"
          href={RAISE_HREF}
          variant="secondary"
          size="small"
        >
          {RAISE_LABEL}
        </Button>
      </section>
    );
  }

  return (
    <section id="experts" className="mb-20 scroll-mt-24">
      <SectionHeader
        titleIcon={<Icon icon={UserAiIcon} size="3rem" aria-hidden />}
        title="Meet the AI Experts"
        subtitle="Hire a ready-made specialist — competent on day one, working for you in minutes."
        actions={
          isLoggedIn ? (
            <div className="flex items-center gap-2">
              <Button
                as="NextLink"
                href="/team"
                variant="secondary"
                size="small"
              >
                View your team
              </Button>
              <Button as="NextLink" href={RAISE_HREF} size="small">
                {RAISE_LABEL}
              </Button>
            </div>
          ) : undefined
        }
      />
      {isLoading ? (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-52 w-full rounded-2xl" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
          {templates.map((template) => (
            <ExpertCard
              key={template.id}
              expert={template}
              category={category}
              isHired={hiredTemplateIds.has(template.id)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
