"use client";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useOnboarding } from "../../../../providers/onboarding/onboarding-provider";
import { resolveResponse } from "@/app/api/helpers";
import { getV1OnboardingState } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { postV2AddMarketplaceAgent } from "@/app/api/__generated__/endpoints/library/library";
import { Confetti } from "@/components/molecules/Confetti/Confetti";
import type { ConfettiRef } from "@/components/molecules/Confetti/Confetti";

export default function Page() {
  const { completeStep } = useOnboarding(7, "AGENT_INPUT");
  const router = useRouter();
  const api = useBackendAPI();
  const [showText, setShowText] = useState(false);
  const [showSubtext, setShowSubtext] = useState(false);
  const confettiRef = useRef<ConfettiRef>(null);

  useEffect(() => {
    // Fire side cannons for a celebratory effect
    const duration = 1500;
    const end = Date.now() + duration;

    function frame() {
      confettiRef.current?.fire({
        particleCount: 4,
        angle: 60,
        spread: 70,
        origin: { x: 0, y: 0.6 },
        shapes: ["square"],
        scalar: 0.8,
        gravity: 0.6,
        decay: 0.93,
      });
      confettiRef.current?.fire({
        particleCount: 4,
        angle: 120,
        spread: 70,
        origin: { x: 1, y: 0.6 },
        shapes: ["square"],
        scalar: 0.8,
        gravity: 0.6,
        decay: 0.93,
      });

      if (Date.now() < end) {
        requestAnimationFrame(frame);
      }
    }

    frame();

    const timer0 = setTimeout(() => {
      setShowText(true);
    }, 100);

    const timer1 = setTimeout(() => {
      setShowSubtext(true);
    }, 500);

    const timer2 = setTimeout(async () => {
      completeStep("CONGRATS");

      try {
        const onboarding = await resolveResponse(getV1OnboardingState());
        if (onboarding?.selectedStoreListingVersionId) {
          try {
            const libraryAgent = await resolveResponse(
              postV2AddMarketplaceAgent({
                store_listing_version_id:
                  onboarding.selectedStoreListingVersionId,
                source: "onboarding",
              }),
            );
            router.replace(`/library/agents/${libraryAgent.id}`);
          } catch (error) {
            console.error("Failed to add agent to library:", error);
            router.replace("/library");
          }
        } else {
          router.replace("/library");
        }
      } catch (error) {
        console.error("Failed to get onboarding data:", error);
        router.replace("/library");
      }
    }, 3000);

    return () => {
      clearTimeout(timer0);
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [completeStep, router, api]);

  return (
    <div className="flex h-screen w-screen flex-col items-center justify-center bg-violet-100">
      <Confetti ref={confettiRef} manualstart />
      <div
        className={cn(
          "z-10 -mb-16 text-9xl duration-500",
          showText ? "opacity-100" : "opacity-0",
        )}
      >
        🎉
      </div>
      <h1
        className={cn(
          "font-poppins text-9xl font-medium tracking-tighter text-violet-700 duration-500",
          showText ? "opacity-100" : "opacity-0",
        )}
      >
        Congrats!
      </h1>
      <p
        className={cn(
          "mb-16 mt-4 font-poppins text-2xl font-medium text-violet-800 transition-opacity duration-500",
          showSubtext ? "opacity-100" : "opacity-0",
        )}
      >
        You earned 3$ for running your first agent
      </p>
    </div>
  );
}
