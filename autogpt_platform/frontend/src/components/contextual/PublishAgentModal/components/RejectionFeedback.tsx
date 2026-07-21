"use client";

import { WarningCircleIcon } from "@phosphor-icons/react";

import { Text } from "@/components/atoms/Text/Text";

interface Props {
  comments: string;
}

export function RejectionFeedback({ comments }: Props) {
  return (
    <div className="mt-4 w-full max-w-md rounded-[14px] border border-rose-200 bg-rose-50 p-3">
      <div className="mb-1 flex items-center gap-2 text-rose-700">
        <WarningCircleIcon size={16} weight="duotone" />
        <Text variant="small-medium" as="span" className="!text-current">
          Review feedback
        </Text>
      </div>
      <Text variant="small" className="text-rose-700">
        {comments}
      </Text>
    </div>
  );
}
