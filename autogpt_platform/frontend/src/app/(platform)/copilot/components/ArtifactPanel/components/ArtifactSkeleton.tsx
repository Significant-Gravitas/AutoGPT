import { Skeleton } from "@/components/ui/skeleton";

interface Props {
  /** Extra line before the 32h block (the variant used while fetching text). */
  extraLine?: boolean;
}

export function ArtifactSkeleton({ extraLine }: Props) {
  return (
    <div className="space-y-3 p-4">
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-4 w-1/2" />
      {extraLine && <Skeleton className="h-4 w-5/6" />}
      <Skeleton className="h-32 w-full" />
    </div>
  );
}
