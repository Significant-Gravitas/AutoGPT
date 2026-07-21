import Avatar, { AvatarImage } from "@/components/atoms/Avatar/Avatar";
import { cn } from "@/lib/utils";

interface Props {
  src?: string;
  name?: string;
  className?: string;
}

export function InitialAvatar({ src, name, className }: Props) {
  const initial = name?.trim().charAt(0).toUpperCase() || "U";

  return (
    <Avatar className={cn("h-10 w-10", className)}>
      <div className="absolute inset-0 z-0 flex items-center justify-center bg-violet-500 text-sm font-semibold text-white">
        {initial}
      </div>
      <AvatarImage
        src={src}
        alt=""
        aria-hidden="true"
        className="relative z-10"
      />
    </Avatar>
  );
}
