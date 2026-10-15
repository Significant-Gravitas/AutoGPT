import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function MemoryPageAccess({ children }: Props) {
  const memory = useFlagStatus(Flag.GRAPHITI_MEMORY);
  const learning = useFlagStatus(Flag.DREAM_SKILL_LEARNING_ENABLED);
  const router = useRouter();
  const allowed =
    (memory.answered && memory.enabled) ||
    (learning.answered && learning.enabled);
  const denied = memory.answered && learning.answered && !allowed;

  useEffect(() => {
    if (denied) router.push("/404");
  }, [denied, router]);

  if (allowed) return children;
  if (denied) return null;
  return (
    <div
      role="status"
      aria-label="Loading memory settings"
      className="flex min-h-screen items-center justify-center"
    >
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
    </div>
  );
}
