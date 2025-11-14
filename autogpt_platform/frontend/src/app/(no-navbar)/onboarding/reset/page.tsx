"use client";
import { postV1ResetOnboardingProgress } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function OnboardingResetPage() {
  const { toast } = useToast();
  const router = useRouter();

  useEffect(() => {
    postV1ResetOnboardingProgress()
      .then(() => {
        toast({
          title: "Onboarding reset successfully",
          description: "You can now start the onboarding process again",
          variant: "success",
        });

        router.push("/onboarding");
      })
      .catch(() => {
        toast({
          title: "Failed to reset onboarding",
          description: "Please try again later",
          variant: "destructive",
        });
      });
  }, [toast, router]);

  return <LoadingSpinner cover />;
}
