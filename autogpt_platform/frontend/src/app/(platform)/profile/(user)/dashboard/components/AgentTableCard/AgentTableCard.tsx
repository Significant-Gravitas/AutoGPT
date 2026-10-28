"use client";

import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import {
  MoreVerticalGlyph,
  StarFilledGlyph,
} from "@/components/atoms/Icon/drawnIcons";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { Status } from "@/components/__legacy__/Status";
import { ImageNotFound01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useImageFallback } from "@/hooks/useImageFallback";

export interface AgentTableCardProps {
  storeAgentSubmission: StoreSubmission;
  onViewSubmission: (submission: StoreSubmission) => void;
}

export const AgentTableCard = ({
  storeAgentSubmission,
  onViewSubmission,
}: AgentTableCardProps) => {
  const onView = () => {
    onViewSubmission(storeAgentSubmission);
  };

  const {
    graph_version,
    name: agentName,
    description,
    image_urls,
    submitted_at,
    status,
    run_count,
    review_avg_rating: rating,
  } = storeAgentSubmission;

  const { showImage, handleImageError } = useImageFallback(image_urls?.[0]);

  return (
    <div
      // TODO(tokens): no exact role (border-neutral-300 #D4D4D4, dark:border-neutral-700)
      className="border-b border-neutral-300 p-4 dark:border-neutral-700"
    >
      <div className="flex gap-4">
        <div
          // TODO(tokens): no exact role (dark:bg-neutral-800)
          className="relative flex aspect-video w-24 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-surface-hover dark:bg-neutral-800"
        >
          {showImage && image_urls?.[0] ? (
            <Image
              src={image_urls[0]}
              unoptimized={isLocalStoreMediaUrl(image_urls[0])}
              alt={agentName}
              fill
              style={{ objectFit: "cover" }}
              onError={handleImageError}
            />
          ) : (
            <Icon
              icon={ImageNotFound01Icon}
              className="h-6 w-6 text-foreground-emphasis"
            />
          )}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3
              // TODO(tokens): no exact role (text-neutral-800 #262626, dark:text-neutral-200)
              className="text-body-md font-medium leading-inherit text-neutral-800 dark:text-neutral-200"
            >
              {agentName}
            </h3>
            <span
              // TODO(tokens): no exact role (text-neutral-500 #737373, dark:text-neutral-400)
              className="text-body-sm leading-inherit text-neutral-500 dark:text-neutral-400"
            >
              v{graph_version}
            </span>
          </div>
          <p
            // TODO(tokens): no exact role (text-neutral-600 #525252, dark:text-neutral-400)
            className="line-clamp-2 text-sm text-neutral-600 dark:text-neutral-400"
          >
            {description}
          </p>
        </div>
        <button
          onClick={onView}
          // TODO(tokens): no exact role (hover:bg-neutral-100 #F5F5F5, dark:hover:bg-neutral-700)
          className="h-fit rounded-full p-1 hover:bg-neutral-100 dark:hover:bg-neutral-700"
        >
          <Icon
            icon={MoreVerticalGlyph}
            aria-label="Icon"
            viewBox="0 0 20 20"
            strokeWidth={1.5}
            // TODO(tokens): no exact role (text-neutral-800 #262626, dark:text-neutral-200)
            className="h-5 w-5 text-neutral-800 dark:text-neutral-200"
          />
        </button>
      </div>

      <div className="mt-4 flex flex-wrap gap-4">
        <Status status={status} />
        <div
          // TODO(tokens): no exact role (text-neutral-600 #525252, dark:text-neutral-400)
          className="text-sm text-neutral-600 dark:text-neutral-400"
        >
          {submitted_at && submitted_at.toLocaleDateString()}
        </div>
        <div
          // TODO(tokens): no exact role (text-neutral-600 #525252, dark:text-neutral-400)
          className="text-sm text-neutral-600 dark:text-neutral-400"
        >
          {(run_count ?? 0).toLocaleString()} runs
        </div>
        <div className="flex items-center gap-1">
          <span
            // TODO(tokens): no exact role (text-neutral-800 #262626, dark:text-neutral-200)
            className="text-sm font-medium text-neutral-800 dark:text-neutral-200"
          >
            {(rating ?? 0).toFixed(1)}
          </span>
          <Icon
            icon={StarFilledGlyph}
            aria-label="Icon"
            // TODO(tokens): no exact role (text-neutral-800 #262626, dark:text-neutral-200)
            className="h-4 w-4 text-neutral-800 dark:text-neutral-200"
          />
        </div>
      </div>
    </div>
  );
};
