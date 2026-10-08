import Link from "next/link";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Home10Icon,
  BubbleChatIcon,
  Folder01Icon,
  GridViewIcon,
  TestTube01Icon,
  ArrowUpRight01Icon,
} from "@hugeicons/core-free-icons";

const links = [
  { label: "Home", href: "/home", icon: Home10Icon },
  { label: "Chat with Otto", href: "/copilot", icon: BubbleChatIcon },
  { label: "My agents", href: "/library", icon: GridViewIcon },
  { label: "Workspace", href: "/artifacts", icon: Folder01Icon },
];

export function LabSidebar() {
  return (
    <aside className="hidden w-52 shrink-0 flex-col border-r border-zinc-200 bg-zinc-50 px-4 py-6 lg:flex">
      <Link href="/" className="mb-10 ml-3 w-fit" aria-label="AutoGPT home">
        <AutoGPTLogo className="h-10 w-24" />
      </Link>
      <nav aria-label="Platform navigation" className="space-y-1">
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-zinc-600 transition-colors hover:bg-zinc-100"
          >
            <Icon icon={link.icon} size={18} />
            {link.label}
          </Link>
        ))}
        <p className="px-3 pb-2 pt-8 text-[10px] font-semibold uppercase tracking-widest text-zinc-400">
          Explore
        </p>
        <Link
          href="/tour/openui"
          aria-current="page"
          className="flex items-center gap-3 rounded-lg bg-purple-50 px-3 py-2.5 text-sm font-medium text-purple-700"
        >
          <Icon icon={TestTube01Icon} size={18} />
          UI Lab
          <span className="ml-auto size-1.5 rounded-full bg-purple-400" />
        </Link>
      </nav>
      <div className="mt-auto rounded-xl border border-zinc-200 bg-white p-4">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-purple-600">
          A little look ahead
        </span>
        <p className="mt-2 text-sm font-medium leading-snug text-zinc-800">
          What if every answer came with the right interface?
        </p>
        <p className="mt-2 text-xs leading-relaxed text-zinc-500">
          An AutoGPT experiment, powered by OpenUI.
        </p>
        <Link
          href="/copilot/openui"
          className="mt-4 flex items-center gap-1 text-xs font-medium text-zinc-700"
        >
          Open in your workspace
          <Icon icon={ArrowUpRight01Icon} size={14} />
        </Link>
      </div>
      <p className="mt-4 px-3 text-[10px] text-zinc-400">
        EXPERIMENT 01 / GENERATIVE UI
      </p>
    </aside>
  );
}
