"use client";

import { useTallyPopup } from "./useTallyPopup";
import { Button } from "@/components/atoms/Button/Button";
import { usePathname, useSearchParams } from "next/navigation";

export function TallyPopupSimple() {
  const { state, handlers } = useTallyPopup();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const isNewBuilder =
    pathname.includes("build") && searchParams.get("view") === "new";

  if (state.isFormVisible) {
    return null;
  }

  if (!state.showTutorial || isNewBuilder) {
    return null;
  }

  return (
    <div className="fixed bottom-1 right-0 z-20 hidden select-none items-center gap-4 p-3 transition-all duration-300 ease-in-out md:flex">
      <Button
        variant="primary"
        onClick={handlers.handleResetTutorial}
        className="mb-0 h-14 w-28 rounded-2xl bg-[rgba(65,65,64,1)] text-left font-sans text-lg font-medium leading-6"
      >
        Tutorial
      </Button>
    </div>
  );
}

export default TallyPopupSimple;
