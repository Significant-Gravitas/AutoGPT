import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout/AuthSplitLayout";
import { SignupMarketingPanel } from "./SignupMarketingPanel";

export function LoadingSignup() {
  return (
    <AuthSplitLayout marketing={<SignupMarketingPanel />}>
      <div className="mb-8 flex flex-col gap-2">
        <Skeleton className="h-8 w-56" />
        <Skeleton className="h-5 w-72" />
      </div>
      <div className="w-full space-y-5">
        <div className="space-y-2">
          <Skeleton className="h-4 w-12" />
          <Skeleton className="h-12 w-full rounded-md" />
        </div>
        <div className="space-y-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-12 w-full rounded-md" />
        </div>
        <div className="space-y-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-12 w-full rounded-md" />
        </div>
        <Skeleton className="h-6 w-full" />
        <Skeleton className="h-12 w-full rounded-md" />
        <div className="flex w-full items-center">
          <Skeleton className="h-px flex-1" />
          <Skeleton className="mx-3 h-4 w-6" />
          <Skeleton className="h-px flex-1" />
        </div>
        <Skeleton className="h-12 w-full rounded-md" />
        <div className="flex justify-center space-x-1">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-12" />
        </div>
      </div>
    </AuthSplitLayout>
  );
}
