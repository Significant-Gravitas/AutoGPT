import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useRouter } from "next/navigation";

export const PublishAuthPrompt = () => {
  const router = useRouter();
  return (
    <div>
      <div className="mx-auto inline-flex h-[370px] w-full flex-col items-center justify-center gap-6 px-4 py-5 sm:px-6">
        <div className="flex flex-col items-center gap-4 text-center">
          <Text variant="h3" className="font-semibold">
            Share your AI creations
          </Text>
          <Text
            variant="lead"
            className="max-w-[80%] text-neutral-600 dark:text-neutral-400"
          >
            Log in or create an account to publish your agents to the
            marketplace and join a community of creators
          </Text>
        </div>
        <div className="flex flex-col items-center gap-3 sm:flex-row">
          <Button
            onClick={() => router.push("/login")}
            className="bg-neutral-800 text-white hover:bg-neutral-900"
          >
            Log in
          </Button>
          <Button onClick={() => router.push("/signup")} variant="secondary">
            Create account
          </Button>
        </div>
      </div>
    </div>
  );
};

export const PublishAuthPromptSkeleton = () => {
  return (
    <div className="mx-auto inline-flex h-[370px] w-full flex-col items-center justify-center gap-6 px-4 py-5 sm:px-6">
      <Skeleton className="h-8 w-64" />
      <Skeleton className="h-20 w-96" />
      <div className="flex flex-col items-center gap-3 sm:flex-row">
        <Skeleton className="h-10 w-24" />
        <Skeleton className="h-10 w-32" />
      </div>
    </div>
  );
};
