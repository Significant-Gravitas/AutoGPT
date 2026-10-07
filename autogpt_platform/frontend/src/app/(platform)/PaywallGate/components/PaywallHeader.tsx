import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { PlayCircleIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";

export function PaywallHeader() {
  return (
    <div className="flex flex-col items-center gap-1 text-center">
      <Text
        variant="h3"
        className="!text-[1.375rem] !leading-[1.6rem] md:!text-[1.75rem] md:!leading-[2.5rem]"
      >
        Your next idea starts here.
      </Text>
      <Text variant="body" className="!text-zinc-500">
        Choose the room you need to bring it to life.
      </Text>
      <Link
        href="/tour/chat?utm_source=platform_paywall"
        target="_blank"
        className="mt-2 inline-flex items-center gap-2 rounded-full border border-violet-200 bg-violet-50/60 px-4 py-1.5 text-sm text-zinc-700 transition-colors hover:bg-violet-100/60"
      >
        <Icon
          icon={PlayCircleIcon}
          className="size-5 shrink-0 text-violet-600"
        />
        <span>Explore a quick demo</span>
      </Link>
    </div>
  );
}
