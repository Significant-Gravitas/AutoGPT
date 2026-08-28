import type { LiveStateHealth } from "@/hooks/useExpertLiveStateHealth";

interface Props {
  health: LiveStateHealth;
}

export function LiveUpdatesStatus({ health }: Props) {
  if (health !== "polling") return null;
  return (
    <p role="status" className="text-xs text-zinc-500">
      Live updates are reconnecting. Progress is refreshing automatically.
    </p>
  );
}
