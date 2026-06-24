import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { cn } from "@/lib/utils";

interface Props {
  src?: string;
  name?: string;
  className?: string;
}

export function InitialAvatar({ src, name, className }: Props) {
  return (
    <Avatar className={cn("h-10 w-10", className)}>
      <AvatarImage src={src} alt={name ? `${name}'s avatar` : "User avatar"} />
      <AvatarFallback>{name?.trim() || "User"}</AvatarFallback>
    </Avatar>
  );
}
