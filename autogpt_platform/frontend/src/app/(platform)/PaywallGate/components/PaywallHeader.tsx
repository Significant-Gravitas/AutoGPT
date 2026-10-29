import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { PlayCircleIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";

export function PaywallHeader() {
  return (
    <div className="flex flex-col items-center gap-1 text-center">
      <Text
        variant="h3"
        className="!text-heading-sm !leading-6.5 md:!text-heading md:!leading-10"
      >
        Choose a plan to keep working
      </Text>
      <Text variant="body" tone="muted">
        Your chats, workflows and files are saved.
      </Text>
      <Link
        href="/tour/chat?utm_source=platform_paywall"
        target="_blank"
        className="mt-2 inline-flex items-center gap-2 rounded-full border border-brand/20 bg-accent-surface/60 px-4 py-1.5 text-sm text-foreground-body transition-colors hover:bg-accent-surface/60"
      >
        <Icon icon={PlayCircleIcon} className="size-5 shrink-0 text-brand" />
        <span>Explore a quick demo</span>
      </Link>
    </div>
  );
}
