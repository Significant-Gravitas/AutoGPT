import { ExpertAvatar } from "../../ExpertAvatar/ExpertAvatar";
import { getManagedIdentity } from "../../ExpertAvatar/helpers";
import { cn } from "@/lib/utils";

interface Props {
  /** The managed looks this Expert may keep: its saved identity, when it has
   *  one, and the General fallback. Never another Expert's face. */
  urls: readonly string[];
  selectedUrl: string;
  disabled: boolean;
  onSelect: (url: string) => void;
}

export function AvatarCatalog({
  urls,
  selectedUrl,
  disabled,
  onSelect,
}: Props) {
  return (
    <div
      role="group"
      aria-label="Managed looks"
      className="flex w-full flex-wrap justify-center gap-2"
    >
      {urls.map((url) => {
        const identity = getManagedIdentity(url);
        const label =
          identity?.visual_category === "general"
            ? "General"
            : (identity?.name ?? "Saved look");
        return (
          <button
            key={url}
            type="button"
            aria-label={label}
            aria-pressed={selectedUrl === url}
            disabled={disabled}
            onClick={() => onSelect(url)}
            className={cn(
              "flex w-24 flex-col items-center gap-1 rounded-lg border border-border p-2 text-xs text-foreground focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50",
              selectedUrl === url && "border-primary bg-muted",
            )}
          >
            <ExpertAvatar name={label} avatarUrl={url} size={40} />
            {label}
          </button>
        );
      })}
    </div>
  );
}
