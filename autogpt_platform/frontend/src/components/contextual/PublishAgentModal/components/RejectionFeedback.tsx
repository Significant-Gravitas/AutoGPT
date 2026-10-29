"use client";
import { Text } from "@/components/atoms/Text/Text";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  comments: string;
}

export function RejectionFeedback({ comments }: Props) {
  return (
    <div className="mt-4 w-full max-w-md rounded-xl border border-error/20 bg-error-surface p-3">
      <div className="mb-1 flex items-center gap-2 text-error-foreground">
        <Icon icon={AlertCircleIcon} size={16} />
        <Text variant="small-medium" as="span" className="!text-current">
          Review feedback
        </Text>
      </div>
      <Text variant="small" className="text-error-foreground">
        {comments}
      </Text>
    </div>
  );
}
