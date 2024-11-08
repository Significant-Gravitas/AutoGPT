"use client";
import React, { useEffect, useState } from "react";
import { Button } from "./ui/button";
import { QuestionMarkCircledIcon } from "@radix-ui/react-icons";
import { useRouter } from "next/navigation";

const TallyPopupSimple = () => {
  const [isFormVisible, setIsFormVisible] = useState(false);
  const router = useRouter();

  useEffect(() => {
    // Load Tally script
    const script = document.createElement("script");
    script.src = "https://tally.so/widgets/embed.js";
    script.async = true;
    document.head.appendChild(script);

    // Setup event listeners for Tally events
    const handleTallyMessage = (event: MessageEvent) => {
      if (typeof event.data === "string") {
        try {
          const data = JSON.parse(event.data);
          if (data.event === "Tally.FormLoaded") {
            setIsFormVisible(true);
          } else if (data.event === "Tally.PopupClosed") {
            setIsFormVisible(false);
          }
        } catch (error) {
          console.error("Error parsing Tally message:", error);
        }
      }
    };

    window.addEventListener("message", handleTallyMessage);

    return () => {
      document.head.removeChild(script);
      window.removeEventListener("message", handleTallyMessage);
    };
  }, []);

  if (isFormVisible) {
    return null; // Hide the button when the form is visible
  }

  const resetTutorial = () => {
    router.push("/build?resetTutorial=true");
  };

  return (
    <div className="fixed bottom-1 right-6 z-50 hidden select-none items-center gap-4 p-3 transition-all duration-300 ease-in-out md:flex">
      <Button
        variant="default"
        onClick={resetTutorial}
        className="font-inter mb-0 h-14 w-28 rounded-2xl bg-[rgba(65,65,64,1)] text-left text-lg font-medium leading-6"
      >
        Tutorial
      </Button>
      <Button
        className="h-14 w-14 rounded-2xl bg-[rgba(65,65,64,1)]"
        variant="default"
        data-tally-open="3yx2L0"
        data-tally-emoji-text="👋"
        data-tally-emoji-animation="wave"
      >
        <QuestionMarkCircledIcon className="h-6 w-6" />
        <span className="sr-only">Reach Out</span>
      </Button>
    </div>
  );
};

export default TallyPopupSimple;
