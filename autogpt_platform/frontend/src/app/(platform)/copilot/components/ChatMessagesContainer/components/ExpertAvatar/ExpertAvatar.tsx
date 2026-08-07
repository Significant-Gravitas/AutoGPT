import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";

interface Props {
  name: string;
  avatarUrl: string | null;
}

export function ExpertAvatar({ name, avatarUrl }: Props) {
  return (
    <Avatar className="h-6 w-6">
      {avatarUrl ? <AvatarImage src={avatarUrl} alt={name} /> : null}
      {/* The fallback seeds a generated avatar off the name, and also covers
          an avatar URL that fails to load. */}
      <AvatarFallback className="h-6 w-6">{name}</AvatarFallback>
    </Avatar>
  );
}
