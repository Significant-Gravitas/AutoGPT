import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { cn } from "@/lib/utils";

interface Props {
  name: string;
  imageUrl: string | null | undefined;
  size?: number;
  className?: string;
}

/**
 * A workflow's own picture, falling back to the same generated marble the
 * expert avatars use. Lets a run row show what ran instead of a glyph that
 * looks the same for every workflow.
 */
export function WorkflowAvatar({
  name,
  imageUrl,
  size = 18,
  className,
}: Props) {
  return (
    <Avatar
      style={{ width: size, height: size }}
      className={cn("shrink-0", className)}
    >
      {imageUrl ? <AvatarImage src={imageUrl} alt={name} /> : null}
      <AvatarFallback>{name}</AvatarFallback>
    </Avatar>
  );
}
