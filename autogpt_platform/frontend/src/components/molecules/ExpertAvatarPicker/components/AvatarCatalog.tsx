import { ExpertAvatar } from "../../ExpertAvatar/ExpertAvatar";
import { EXPERT_AVATARS } from "../../ExpertAvatar/helpers";
import { cn } from "@/lib/utils";

interface Props {
  selectedUrl: string;
  disabled: boolean;
  onSelect: (id: string) => void;
}

export function AvatarCatalog({ selectedUrl, disabled, onSelect }: Props) {
  return (
    <div
      role="group"
      aria-label="Avatar catalog"
      className="grid w-full grid-cols-4 gap-2"
    >
      {EXPERT_AVATARS.map((avatar) => (
        <button
          key={avatar.id}
          type="button"
          aria-label={avatar.label}
          aria-pressed={selectedUrl === avatar.url}
          disabled={disabled}
          onClick={() => onSelect(avatar.id)}
          className={cn(
            "flex flex-col items-center gap-1 rounded-lg border border-border p-2 text-xs text-foreground focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50",
            selectedUrl === avatar.url && "border-primary bg-muted",
          )}
        >
          <ExpertAvatar name={avatar.label} avatarUrl={avatar.url} size={40} />
          {avatar.label}
        </button>
      ))}
    </div>
  );
}
