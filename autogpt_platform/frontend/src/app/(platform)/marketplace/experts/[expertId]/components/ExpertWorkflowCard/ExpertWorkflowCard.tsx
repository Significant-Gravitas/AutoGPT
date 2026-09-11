"use client";

import type { ExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { FlashIcon } from "@hugeicons/core-free-icons";
import Image from "next/image";
import { useExpertWorkflowCard } from "./useExpertWorkflowCard";

interface Props {
  workflow: ExpertWorkflowRef;
  accent: ExpertAccent;
}

export function ExpertWorkflowCard({ workflow, accent }: Props) {
  const { imageUrl, isLoadingImage } = useExpertWorkflowCard(
    workflow.store_listing_version_id,
  );
  const name = workflow.name ?? "Unnamed workflow";

  return (
    <li className="flex flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white">
      <div className="relative aspect-[2.17/1] w-full bg-zinc-50">
        {imageUrl ? (
          <Image
            src={imageUrl}
            alt={`${name} preview image`}
            fill
            sizes="(min-width: 640px) 360px, 100vw"
            className="object-cover"
          />
        ) : isLoadingImage ? (
          <Skeleton className="absolute inset-0 rounded-none" />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-white ring-1 ring-inset ring-zinc-200/70">
              <Icon icon={FlashIcon} size={18} className={accent.icon} />
            </span>
          </div>
        )}
      </div>
      <div className="flex flex-1 flex-col border-t border-zinc-100 p-4">
        <div className="text-sm font-medium text-zinc-900">{name}</div>
        {workflow.description ? (
          <p className="mt-1 line-clamp-2 text-[13px] leading-5 text-zinc-500">
            {workflow.description}
          </p>
        ) : null}
      </div>
    </li>
  );
}
