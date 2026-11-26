import { cn } from "@/lib/utils";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

const FLANK = [
  {
    name: "Maya",
    avatarUrl: "/autogpt-characters/v2.1/expert-maya/neutral/128.webp",
    className: "left-0 top-1 -rotate-6",
  },
  {
    name: "Devon",
    avatarUrl: "/autogpt-characters/v2.1/expert-devon/neutral/128.webp",
    className: "right-0 top-1 rotate-6",
  },
];

/** Otto flanked by two experts, as on the Team page header. */
export function TeamAvatars() {
  return (
    <div aria-hidden className="relative h-14 w-28 shrink-0">
      {FLANK.map((expert) => (
        <span key={expert.name} className={cn("absolute", expert.className)}>
          <ExpertAvatar
            name={expert.name}
            avatarUrl={expert.avatarUrl}
            size={44}
            frame="stacked"
          />
        </span>
      ))}
      <span className="absolute top-0 left-1/2 z-raised -translate-x-1/2">
        <AutopilotAvatar size={52} frame="stacked" />
      </span>
    </div>
  );
}
