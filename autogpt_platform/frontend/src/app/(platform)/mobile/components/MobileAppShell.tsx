import { Bell, ChatsCircle, Gear, Users } from "@phosphor-icons/react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { CSSProperties, ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  children: ReactNode;
}

const destinations = [
  { id: "chats", label: "Chats", href: "/mobile?tab=chats", icon: ChatsCircle },
  { id: "experts", label: "Experts", href: "/mobile?tab=experts", icon: Users },
  {
    id: "attention",
    label: "Needs you",
    href: "/mobile?tab=attention",
    icon: Bell,
  },
  { id: "settings", label: "Settings", href: "/settings/profile", icon: Gear },
] as const;

export function MobileAppShell({ children }: Props) {
  const pathname = usePathname();
  const params = useSearchParams();
  const activeTab = pathname.startsWith("/settings")
    ? "settings"
    : pathname.startsWith("/marketplace")
      ? "experts"
      : pathname === "/mobile"
        ? (params.get("tab") ?? "chats")
        : "chats";

  return (
    <main
      className="flex h-dvh min-w-0 flex-col bg-white [&_input]:text-base [&_textarea]:text-base"
      style={{ "--mobile-navigation-height": "64px" } as CSSProperties}
    >
      <section className="min-h-0 flex-1 overflow-auto">{children}</section>
      <nav
        aria-label="App navigation"
        className="grid h-16 shrink-0 grid-cols-4 border-t border-zinc-200 bg-white"
      >
        {destinations.map(({ id, label, href, icon: Icon }) => (
          <Link
            key={id}
            href={href}
            aria-current={activeTab === id ? "page" : undefined}
            className={cn(
              "flex min-w-0 flex-col items-center justify-center gap-1 text-xs font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-4px]",
              activeTab === id ? "text-zinc-950" : "text-zinc-500",
            )}
          >
            <Icon
              size={22}
              weight={activeTab === id ? "fill" : "regular"}
              aria-hidden
            />
            {label}
          </Link>
        ))}
      </nav>
    </main>
  );
}
