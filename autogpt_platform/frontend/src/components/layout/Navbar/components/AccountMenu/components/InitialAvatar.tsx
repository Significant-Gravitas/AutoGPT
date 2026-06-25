import Avatar, {
  AvatarImage,
  AvatarFallback,
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
      <AvatarFallback>{name}</AvatarFallback>
      <AvatarImage
        src={src}
        alt=""
        aria-hidden="true"
        className="relative z-10"
      />
    </Avatar>
  );
}
