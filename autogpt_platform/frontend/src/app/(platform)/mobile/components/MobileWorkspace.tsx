import Link from "next/link";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { NativePushControl } from "@/services/push-notifications/native/NativePushControl";
import { MobileChats } from "./MobileChats";
import { MobileExperts } from "./MobileExperts";
import { MobileAttention } from "./MobileAttention";

interface Props {
  tab: "chats" | "experts" | "attention";
}

const tabs = [
  { id: "chats", label: "Chats" },
  { id: "experts", label: "Experts" },
  { id: "attention", label: "Needs you" },
] as const;

export function MobileWorkspace({ tab }: Props) {
  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 px-4 pb-12 pt-6 sm:px-6">
      <header className="flex flex-col gap-2">
        <Text variant="h2">Your workspace</Text>
        <Text variant="body" tone="secondary">
          Talk with your experts and keep work moving.
        </Text>
      </header>
      <nav
        aria-label="Mobile workspace"
        className="grid grid-cols-3 gap-1 rounded-full border border-zinc-200 bg-white p-1"
      >
        {tabs.map((item) => (
          <Link
            key={item.id}
            href={`/mobile?tab=${item.id}`}
            aria-current={tab === item.id ? "page" : undefined}
            className={cn(
              "flex min-h-11 items-center justify-center rounded-full px-3 text-sm font-medium focus-visible:outline focus-visible:outline-2",
              tab === item.id
                ? "bg-zinc-900 text-white"
                : "text-zinc-600 hover:bg-zinc-100",
            )}
          >
            {item.label}
          </Link>
        ))}
      </nav>
      <NativePushControl />
      {tab === "chats" && <MobileChats />}
      {tab === "experts" && <MobileExperts />}
      {tab === "attention" && <MobileAttention />}
    </div>
  );
}
