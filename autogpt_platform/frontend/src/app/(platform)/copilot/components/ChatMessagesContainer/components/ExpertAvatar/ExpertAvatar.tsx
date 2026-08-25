import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";

interface Props {
  name: string;
  avatarUrl: string | null;
  isAutopilot?: boolean;
}

export function ExpertAvatar({ name, avatarUrl, isAutopilot }: Props) {
  if (isAutopilot && !avatarUrl) {
    return (
      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-b from-white to-zinc-100 shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),inset_0_-2px_4px_rgba(0,0,0,0.08),0_1px_2px_rgba(0,0,0,0.06)] ring-1 ring-inset ring-zinc-200/70">
        <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-5" />
      </span>
    );
  }

  return (
    <Avatar className="h-9 w-9">
      {avatarUrl ? <AvatarImage src={avatarUrl} alt={name} /> : null}
      {/* The fallback seeds a generated avatar off the name, and also covers
          an avatar URL that fails to load. */}
      <AvatarFallback className="h-9 w-9">{name}</AvatarFallback>
    </Avatar>
  );
}
