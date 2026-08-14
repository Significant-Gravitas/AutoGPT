"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { AITeamIcon } from "@/components/atoms/AITeamIcon/AITeamIcon";
import Link from "next/link";
import { SectionHeader } from "../SectionHeader";
import { ExpertCard } from "./components/ExpertCard";
import { ExpertProfileSheet } from "./components/ExpertProfileSheet/ExpertProfileSheet";
import { useExpertsSection } from "./useExpertsSection";

export function ExpertsSection() {
  const {
    isLoggedIn,
    templates,
    hiredTemplateIds,
    isLoading,
    isError,
    selectedTemplateId,
    openTemplate,
    closeSheet,
  } = useExpertsSection();

  if (!isLoggedIn) {
    return null;
  }

  if (isError || (!isLoading && templates.length === 0)) {
    // Raising an expert needs no roster templates, so the second door
    // stays open even when the template list is empty or failed to load.
    return (
      <section id="experts" className="mb-20">
        <RaiseLink />
      </section>
    );
  }

  return (
    <section id="experts" className="mb-20">
      <SectionHeader
        titleIcon={<AITeamIcon size={30} />}
        title="Meet the AI Experts"
        subtitle="Hire a ready-made specialist — competent on day one, working for you in minutes."
        action={{ label: "View your team", href: "/team" }}
      />
      <div className="-mt-3 mb-6">
        <RaiseLink />
      </div>
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

function RaiseLink() {
  return (
    <Link
      href="/raise"
      className="text-sm font-medium text-purple-600 transition-colors hover:text-purple-700"
    >
      …or raise your own expert from scratch
    </Link>
  );
}
