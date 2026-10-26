import { Icon } from "@/components/atoms/Icon/Icon";
import { LinkSquare02Icon } from "@hugeicons/core-free-icons";

interface Props {
  providerName: string;
  /** The vendor's page for creating or copying a key, if the provider
   *  declares one: a first-time user rarely knows where it lives. */
  href?: string | null;
}

export function ApiKeyHelpLink({ providerName, href }: Props) {
  if (!href) return null;
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex w-fit items-center gap-1 rounded-sm text-sm font-medium text-foreground-emphasis underline underline-offset-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-border-strong"
    >
      Where do I find my {providerName} key?
      <Icon icon={LinkSquare02Icon} size={14} aria-hidden />
      <span className="sr-only">(opens in a new tab)</span>
    </a>
  );
}
