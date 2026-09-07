import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { cn } from "@/lib/utils";

/** The two sizes the home surfaces draw: a row bullet and a task marker. */
type WorkflowAvatarSize = 18 | 36;

const SIZE_CLASS: Record<WorkflowAvatarSize, string> = {
  18: "size-[18px]",
  36: "size-9",
};

interface Props {
  name: string;
  imageUrl: string | null | undefined;
  size?: WorkflowAvatarSize;
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
    <Avatar className={cn("shrink-0", SIZE_CLASS[size], className)}>
      {imageUrl ? <AvatarImage src={imageUrl} alt={name} /> : null}
      <AvatarFallback>{name}</AvatarFallback>
    </Avatar>
  );
}
