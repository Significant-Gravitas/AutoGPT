import { useState } from "react";
import { createRoot } from "react-dom/client";
import { RecommendedExpertCard } from "@/app/(no-navbar)/onboarding/steps/HireStep/components/RecommendedExpertCard";
import "@/app/globals.css";

function Fixture() {
  const [isIncluded, setIsIncluded] = useState(true);
  return (
    <div className="relative mx-auto mt-40 w-72">
      {new URLSearchParams(location.search).has("covered") ? (
        <div
          data-testid="overlay"
          className="pointer-events-none absolute -inset-8 z-10 bg-gray-100"
        />
      ) : null}
      <RecommendedExpertCard
        expert={{
          template_id: "maria",
          name: "Maria",
          role: "marketing",
          reason: "Plans and runs your campaigns.",
        }}
        isHired={false}
        isHiringThis={false}
        isIncluded={isIncluded}
        disabled={false}
        onHire={() => {}}
        onToggleIncluded={() => setIsIncluded(!isIncluded)}
      />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<Fixture />);
