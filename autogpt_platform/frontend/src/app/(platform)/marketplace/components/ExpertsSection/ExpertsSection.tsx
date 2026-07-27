"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { UsersThreeIcon } from "@phosphor-icons/react";
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
    <section className="mb-12">
      <div className="mb-8 flex flex-row items-center gap-2">
        <UsersThreeIcon size={24} />
        <Text variant="h4">Meet the Experts</Text>
      </div>
      {isLoading ? (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-40 w-full rounded-2xl" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
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
        templateId={selectedTemplateId}
        onClose={closeSheet}
      />
    </section>
  );
}
