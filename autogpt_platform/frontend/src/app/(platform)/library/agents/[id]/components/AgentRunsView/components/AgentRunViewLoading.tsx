import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { Skeleton } from "@/components/ui/skeleton";

export function AgentRunViewLoading() {
  return (
    <div className="flex flex-col gap-4">
      <Breadcrumbs
        items={[
          { name: "My Library", link: "/library" },
          { name: "Agent Runs", link: `/library/agents/*****` },
        ]}
      />
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-10 w-full" />
    </div>
  );
}
