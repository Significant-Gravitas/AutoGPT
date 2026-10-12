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
        Choose the plan that&apos;s right for{" "}
        <span className="bg-gradient-to-r from-purple-500 to-indigo-500 bg-clip-text text-transparent">
          you
        </span>
      </Text>
      <Text variant="body" className="!text-zinc-500">
        Pick a plan to start working with experts and running agents.
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
        <span>
          Not sure yet?{" "}
          <span className="font-semibold text-violet-700">Try it</span> —
          Instant demo — No signup
        </span>
      </Link>
    </div>
  );
}
