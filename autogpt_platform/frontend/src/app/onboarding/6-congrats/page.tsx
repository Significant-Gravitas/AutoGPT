"use client";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { finishOnboarding } from "./actions";
import confetti from "canvas-confetti";

export default function Page() {
  const [showText, setShowText] = useState(false);
  const [showSubtext, setShowSubtext] = useState(false);

  useEffect(() => {
    confetti({
      particleCount: 100,
      spread: 360,
      shapes: ["square", "circle"],
      scalar: 2,
      decay: 0.92,
      origin: { y: 0.4, x: 0.5 },
    });

    const timer0 = setTimeout(() => {
      setShowText(true);
    }, 100);

    const timer1 = setTimeout(() => {
      setShowSubtext(true);
    }, 500);

    const timer2 = setTimeout(() => {
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
        You earned 10$ for running your first agent
      </p>
    </div>
  );
}
