import { cn } from "@/lib/utils";
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
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={notionAvatarUrlFor(config)}
      alt={title ?? "Avatar"}
      width={size}
      height={size}
      loading="lazy"
      decoding="async"
      data-testid="notion-avatar-image"
      data-avatar={notionAvatarUrlFor(config)}
      className={cn("shrink-0 rounded-full", className)}
      style={{ width: size, height: size }}
    />
  );
}
