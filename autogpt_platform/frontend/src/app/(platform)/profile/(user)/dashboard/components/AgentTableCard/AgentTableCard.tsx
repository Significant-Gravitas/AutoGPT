"use client";

import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { SubmissionStatusBadge } from "../SubmissionStatusBadge";
import {
  ImageNotFound01Icon,
  MoreVerticalIcon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
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
    <div className="border-b border-zinc-300 p-4">
      <div className="flex gap-4">
        <div className="relative flex aspect-video w-24 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-zinc-100">
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
              className="h-6 w-6 text-zinc-800"
            />
          )}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <Text
              variant="body-medium"
              as="h3"
              tone="primary"
              className="text-[15px]"
              unmask={false}
            >
              {agentName}
            </Text>
            <Text
              variant="small"
              as="span"
              tone="muted"
              className="text-[13px]"
            >
              v{graph_version}
            </Text>
          </div>
          <Text
            variant="body"
            tone="secondary"
            className="line-clamp-2"
            unmask={false}
          >
            {description}
          </Text>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="View submission"
          withTooltip={false}
          onClick={onView}
          className="rounded-full"
          leftIcon={
            <Icon icon={MoreVerticalIcon} size={20} className="text-zinc-800" />
          }
        />
      </div>

      <div className="mt-4 flex flex-wrap gap-4">
        <SubmissionStatusBadge status={status} />
        <div className="text-sm text-zinc-600">
          {submitted_at && submitted_at.toLocaleDateString()}
        </div>
        <div className="text-sm text-zinc-600">
          {(run_count ?? 0).toLocaleString()} runs
        </div>
        <div className="flex items-center gap-1">
          <Text variant="body-medium" as="span" tone="primary">
            {(rating ?? 0).toFixed(1)}
          </Text>
          <Icon icon={StarIcon} size={16} className="text-zinc-800" />
        </div>
      </div>
    </div>
  );
};
