"use client";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { finishOnboarding } from "./actions";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import * as party from "party-js";

export default function Page() {
  const { state, updateState } = useOnboarding(7, "AGENT_INPUT");
  const [showText, setShowText] = useState(false);
  const [showSubtext, setShowSubtext] = useState(false);
  const divRef = useRef(null);

  useEffect(() => {
    if (divRef.current) {
      party.confetti(divRef.current, {
        count: 100,
        spread: 180,
        shapes: ["square", "circle"],
        size: party.variation.range(2, 2), // scalar: 2
        speed: party.variation.range(300, 1000),
      });
    }

    const timer0 = setTimeout(() => {
      setShowText(true);
    }, 100);

    const timer1 = setTimeout(() => {
      setShowSubtext(true);
    }, 500);

    const timer2 = setTimeout(() => {
      updateState({
        completedSteps: [...(state?.completedSteps || []), "CONGRATS"],
      });
      finishOnboarding();
    }, 3000);

    return () => {
      clearTimeout(timer0);
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  return (
    <div className="flex h-screen w-screen flex-col items-center justify-center bg-violet-100">
      <div
        ref={divRef}
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
