"use client";

import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import {
  MoreVerticalGlyph,
  StarFilledGlyph,
} from "@/components/atoms/Icon/drawnIcons";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { SubmissionStatus } from "../SubmissionStatus/SubmissionStatus";
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
    <div className="border-b border-border-strong p-4">
      <div className="flex gap-4">
        <div className="relative flex aspect-video w-24 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-surface-hover text-foreground-emphasis">
          {showImage && image_urls?.[0] ? (
            <Image
              src={image_urls[0]}
              unoptimized={isLocalStoreMediaUrl(image_urls[0])}
              alt={agentName}
              fill
              className="object-cover"
              onError={handleImageError}
            />
          ) : (
            <Icon icon={ImageNotFound01Icon} className="h-6 w-6" />
          )}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="text-body-md font-medium leading-inherit text-foreground-emphasis">
              {agentName}
            </h3>
            <span className="text-body-sm leading-inherit text-foreground-tertiary">
              v{graph_version}
            </span>
          </div>
          <p className="line-clamp-2 text-sm text-foreground-secondary">
            {description}
          </p>
        </div>
        <button
          onClick={onView}
          className="h-fit rounded-full p-1 text-foreground-emphasis hover:bg-surface-hover"
        >
          <Icon
            icon={MoreVerticalGlyph}
            aria-label="Icon"
            viewBox="0 0 20 20"
            strokeWidth={1.5}
            className="h-5 w-5"
          />
        </button>
      </div>

      <div className="mt-4 flex flex-wrap gap-4">
        <SubmissionStatus status={status} />
        <div className="text-sm text-foreground-secondary">
          {submitted_at && submitted_at.toLocaleDateString()}
        </div>
        <div className="text-sm text-foreground-secondary">
          {(run_count ?? 0).toLocaleString()} runs
        </div>
        <div className="flex items-center gap-1 text-foreground-emphasis">
          <span className="text-sm font-medium text-foreground-emphasis">
            {(rating ?? 0).toFixed(1)}
          </span>
          <Icon icon={StarFilledGlyph} aria-label="Icon" className="h-4 w-4" />
        </div>
      </div>
    </div>
  );
};
