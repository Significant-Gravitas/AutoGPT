import { Text } from "@/components/atoms/Text/Text";
import { BulbIcon, SquareIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function StoppedTaskCard() {
  return (
    <div className="my-2 flex animate-in items-start gap-3 rounded-xl border border-zinc-200/70 bg-white p-4 shadow-xs duration-500 fill-mode-both fade-in slide-in-from-bottom-2">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-purple-50">
        <Icon icon={SquareIcon} size={16} className="text-purple-500" />
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
          <Icon
            icon={BulbIcon}
            size={14}
            className="shrink-0 text-purple-300"
          />
          <Text variant="small" className="text-zinc-600">
            Try &ldquo;continue&rdquo; or type something new.
          </Text>
        </div>
      </div>
    </div>
  );
}
