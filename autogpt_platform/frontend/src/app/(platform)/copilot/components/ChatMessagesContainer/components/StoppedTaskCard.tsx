import { LightbulbIcon, SquareIcon } from "@phosphor-icons/react";

import { Text } from "@/components/atoms/Text/Text";

export function StoppedTaskCard() {
  return (
    <div className="my-2 flex items-start gap-3 rounded-medium border border-zinc-200 bg-zinc-50 p-4">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-small bg-purple-50">
        <SquareIcon size={16} weight="fill" className="text-purple-500" />
      </div>
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" className="text-zinc-900">
          Task stopped
        </Text>
        <Text variant="body" className="mt-1 text-[13px] text-zinc-600">
          The response above is incomplete. You can ask to continue or type
          something new.
        </Text>
        <div className="mt-2.5 flex items-center gap-1.5">
          <LightbulbIcon size={14} className="shrink-0 text-purple-300" />
          <Text variant="small" className="text-zinc-600">
            Try &ldquo;continue&rdquo; or type something new.
          </Text>
        </div>
      </div>
    </div>
  );
}
