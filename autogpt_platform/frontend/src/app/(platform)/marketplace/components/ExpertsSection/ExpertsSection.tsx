"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { AITeamIcon } from "@/components/atoms/AITeamIcon/AITeamIcon";
import { SectionHeader } from "../SectionHeader";
import { ExpertCard } from "./components/ExpertCard";
import { ExpertProfileSheet } from "./components/ExpertProfileSheet/ExpertProfileSheet";
import { useExpertsSection } from "./useExpertsSection";

export function ExpertsSection() {
  const {
    templates,
    hiredTemplateIds,
    isLoading,
    isError,
    selectedTemplateId,
    openTemplate,
    closeSheet,
  } = useExpertsSection();

  if (isError || (!isLoading && templates.length === 0)) {
    return null;
  }

  return (
    <section className="mb-20">
      <SectionHeader
        titleIcon={<AITeamIcon size={30} />}
        title="Meet the AI Experts"
        subtitle="Hire a ready-made specialist — competent on day one, working for you in minutes."
        action={{ label: "View your team", href: "/team" }}
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
              onClick={() => openTemplate(template.id)}
            />
          ))}
        </div>
      )}
      <ExpertProfileSheet
        expert={templates.find((t) => t.id === selectedTemplateId) ?? null}
        onClose={closeSheet}
      />
    </section>
  );
}
