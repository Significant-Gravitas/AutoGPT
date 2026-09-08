import { Text } from "@/components/atoms/Text/Text";
import {
  STATUSES,
  type AvatarStatus,
} from "@/components/molecules/BotAvatar/helpers";
import { cn } from "@/lib/utils";

interface Props {
  status: AvatarStatus;
  onChange: (status: AvatarStatus) => void;
}

export function StatusToggle({ status, onChange }: Props) {
  const current = STATUSES.find((option) => option.id === status);
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        role="radiogroup"
        aria-label="Status"
        className="flex rounded-full bg-zinc-100 p-1"
      >
        {STATUSES.map((option) => {
          const isSelected = option.id === status;
          return (
            <button
              key={option.id}
              type="button"
              role="radio"
              aria-checked={isSelected}
              onClick={() => onChange(option.id)}
              className={cn(
                "rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300",
                isSelected
                  ? "bg-white text-zinc-900 shadow-subtle"
                  : "text-zinc-500 hover:text-zinc-800",
              )}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      <Text variant="small" as="p" className="text-zinc-500">
        {current?.hint}
      </Text>
    </div>
  );
}
