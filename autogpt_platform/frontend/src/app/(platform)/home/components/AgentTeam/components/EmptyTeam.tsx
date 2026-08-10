import Link from "next/link";
import { Text } from "@/components/atoms/Text/Text";

export function EmptyTeam() {
  return (
    <div className="rounded-lg border border-dashed border-zinc-200 p-5 text-center">
      <Text variant="body-medium">Build your team</Text>
      <Text variant="small" className="mt-1 text-zinc-500">
        Hire an expert to start delegating work.
      </Text>
      <Link
        href="/marketplace"
        className="mt-3 inline-block text-sm font-medium text-zinc-700 hover:text-zinc-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
      >
        Browse experts
      </Link>
    </div>
  );
}
