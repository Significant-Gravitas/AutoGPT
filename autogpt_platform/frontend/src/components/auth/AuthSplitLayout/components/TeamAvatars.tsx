import { cn } from "@/lib/utils";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

const AVATAR_CLASS =
  "rounded-full border border-[#e3e3e3] ring-2 ring-[#f9f9f9]";

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
            className={AVATAR_CLASS}
          />
        </span>
      ))}
      <span className="absolute left-1/2 top-0 z-10 -translate-x-1/2">
        <AutopilotAvatar size={52} className={AVATAR_CLASS} />
      </span>
    </div>
  );
}
