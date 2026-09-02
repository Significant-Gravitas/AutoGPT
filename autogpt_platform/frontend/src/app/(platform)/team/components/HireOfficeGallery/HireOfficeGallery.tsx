"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Building06Icon } from "@hugeicons/core-free-icons";
import { OfficeDialog } from "./components/OfficeDialog";
import { OfficePackCard } from "./components/OfficePackCard";
import { useHireOfficeGallery } from "./useHireOfficeGallery";

export function HireOfficeGallery() {
  const {
    templates,
    isLoading,
    isError,
    refetch,
    selectedTemplate,
    openPreview,
    closePreview,
    hire,
    isHiring,
    hireResult,
  } = useHireOfficeGallery();

  if (isError) {
    return (
      <ErrorCard
        context="office packs"
        hint="We could not load the offices you can hire."
        onRetry={() => refetch()}
      />
    );
  }

  if (!isLoading && templates.length === 0) {
    return null;
  }

  return (
    <section aria-label="Hire an office" className="space-y-4">
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <Icon icon={Building06Icon} size={18} className="text-zinc-950" />
          <Text variant="h5">Hire an office</Text>
        </div>
        <Text variant="body" className="max-w-prose text-zinc-600">
          Bring on a ready-made team of experts in one go — schedules and first
          tasks included.
        </Text>
      </div>

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-44 rounded-3xl" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-3">
          {templates.map((template) => (
            <OfficePackCard
              key={template.id}
              template={template}
              onSelect={openPreview}
            />
          ))}
        </div>
      )}

      <OfficeDialog
        template={selectedTemplate}
        hireResult={hireResult}
        isHiring={isHiring}
        onHire={hire}
        onClose={closePreview}
      />
    </section>
  );
}
