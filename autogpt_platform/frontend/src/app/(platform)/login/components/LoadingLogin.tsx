import { AuthCard } from "@/components/auth/AuthCard";
import { Skeleton } from "@/components/ui/skeleton";

export function LoadingLogin() {
  return (
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center">
      <AuthCard title="">
        <div className="w-full space-y-6">
          <Skeleton className="mx-auto h-8 w-48" />
          <Skeleton className="h-12 w-full rounded-md" />
          <div className="flex w-full items-center">
            <Skeleton className="h-px flex-1" />
            <Skeleton className="mx-3 h-4 w-6" />
            <Skeleton className="h-px flex-1" />
          </div>
          <div className="space-y-2">
            <Skeleton className="h-4 w-12" />
            <Skeleton className="h-12 w-full rounded-md" />
          </div>
          <div className="space-y-2">
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-12 w-full rounded-md" />
          </div>
          <Skeleton className="h-16 w-full rounded-md" />
          <Skeleton className="h-12 w-full rounded-md" />
          <div className="flex justify-center space-x-1">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-4 w-12" />
          </div>
        </div>
      </AuthCard>
    </div>
  );
}
