"use client";
import React, { useEffect, useState } from "react";
import { Button } from "./ui/button";
import { IconMegaphone } from "@/components/ui/icons";

const TallyPopupSimple = () => {
  const [isFormVisible, setIsFormVisible] = useState(false);

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
    const url = `${window.location.origin}/build?resetTutorial=true`;
    window.location.href = url;
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 flex items-center gap-4 p-3 transition-all duration-300 ease-in-out">
      <Button variant="default" onClick={resetTutorial} className="mb-0">
        Tutorial
      </Button>
      <Button
        variant="default"
        data-tally-open="3yx2L0"
        data-tally-emoji-text="👋"
        data-tally-emoji-animation="wave"
      >
        <IconMegaphone size="lg" />
        <span className="sr-only">Reach Out</span>
      </Button>
    </div>
  );
};

export default TallyPopupSimple;
