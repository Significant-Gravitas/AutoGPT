"use client";
import { Text } from "@/components/atoms/Text/Text";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  comments: string;
}

export function RejectionFeedback({ comments }: Props) {
  return (
    <div
      // TODO(tokens): no exact role for border-rose-200 #fecdd3, bg-rose-50 #fff1f2
      className="mt-4 w-full max-w-md rounded-[14px] border border-rose-200 bg-rose-50 p-3"
    >
      <div
        // TODO(tokens): no exact role for text-rose-700 #be123c
        className="mb-1 flex items-center gap-2 text-rose-700"
      >
        <Icon icon={AlertCircleIcon} size={16} />
        <Text variant="small-medium" as="span" className="!text-current">
          Review feedback
        </Text>
      </div>
      <Text
        // TODO(tokens): no exact role for text-rose-700 #be123c
        variant="small"
        className="text-rose-700"
      >
        {comments}
      </Text>
    </div>
  );
}
