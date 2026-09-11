"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { AITeamIcon } from "@/components/atoms/AITeamIcon/AITeamIcon";
import { Button } from "@/components/atoms/Button/Button";
import { SectionHeader } from "../SectionHeader";
import { ExpertCard } from "./components/ExpertCard";
import { useExpertsSection } from "./useExpertsSection";

const RAISE_LABEL = "Raise your own";
const RAISE_HREF = "/raise";

export function ExpertsSection() {
  const { isLoggedIn, templates, hiredTemplateIds, isLoading, isError } =
    useExpertsSection();

  if (isError || (!isLoading && templates.length === 0)) {
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
        titleIcon={<AITeamIcon size={30} />}
        title="Meet the AI Experts"
        subtitle="Hire a ready-made specialist — competent on day one, working for you in minutes."
        action={
          isLoggedIn ? { label: "View your team", href: "/team" } : undefined
        }
        secondaryAction={
          isLoggedIn ? { label: RAISE_LABEL, href: RAISE_HREF } : undefined
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
              isHired={hiredTemplateIds.has(template.id)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
