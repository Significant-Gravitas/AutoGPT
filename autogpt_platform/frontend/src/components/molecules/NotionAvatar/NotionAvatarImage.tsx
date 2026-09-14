import { cn } from "@/lib/utils";
import Image from "next/image";
import { notionAvatarUrlFor, type NotionAvatarConfig } from "./helpers";

interface Props {
  config: NotionAvatarConfig;
  size?: number;
  title?: string;
  className?: string;
}

/** The avatar at rest, as a cacheable image. Costs no JavaScript, so it is the
 *  right call for lists and rows where nothing is animating anyway. */
export function NotionAvatarImage({
  config,
  size = 36,
  title,
  className,
}: Props) {
  const src = notionAvatarUrlFor(config);

  return (
    <Image
      src={src}
      alt={title ?? "Avatar"}
      width={size}
      height={size}
      unoptimized
      data-testid="notion-avatar-image"
      data-avatar={src}
      className={cn("shrink-0 rounded-full", className)}
    />
  );
}
