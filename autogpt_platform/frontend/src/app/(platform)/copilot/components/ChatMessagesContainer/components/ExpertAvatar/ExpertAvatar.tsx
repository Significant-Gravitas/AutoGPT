import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";

interface Props {
  name: string;
  avatarUrl: string | null;
  size?: "default" | "small";
}

export function ExpertAvatar({ name, avatarUrl, size = "default" }: Props) {
  const dimension = size === "small" ? "h-4 w-4" : "h-6 w-6";
  return (
    <Avatar className={dimension}>
      {avatarUrl ? <AvatarImage src={avatarUrl} alt={name} /> : null}
      {/* The fallback seeds a generated avatar off the name, and also covers
          an avatar URL that fails to load. */}
      <AvatarFallback className={dimension}>{name}</AvatarFallback>
    </Avatar>
  );
}
