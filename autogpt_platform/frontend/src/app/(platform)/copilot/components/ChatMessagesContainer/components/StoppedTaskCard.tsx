import { Text } from "@/components/atoms/Text/Text";
import { BulbIcon, SquareIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function StoppedTaskCard() {
  return (
    <div className="my-2 flex items-start gap-3 rounded-xl border border-border-base/70 bg-surface p-4 shadow-sm duration-500 animate-in fade-in fill-mode-both slide-in-from-bottom-2">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-small bg-accent-surface">
        <Icon icon={SquareIcon} size={16} className="text-brand" />
      </div>
      <div className="min-w-0 flex-1">
        <Text tone="primary" variant="body-medium">
          Task stopped
        </Text>
        <Text
          tone="secondary"
          variant="body"
          className="mt-1 text-body-sm leading-inherit"
        >
          The response above is incomplete. You can ask to continue or type
          something new.
        </Text>
        <div className="mt-2.5 flex items-center gap-1.5">
          <Icon icon={BulbIcon} size={14} className="shrink-0 text-brand" />
          <Text tone="secondary" variant="small">
            Try &ldquo;continue&rdquo; or type something new.
          </Text>
        </div>
      </div>
    </div>
  );
}
