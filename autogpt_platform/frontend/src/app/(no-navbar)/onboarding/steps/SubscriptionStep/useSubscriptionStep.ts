import { useState } from "react";
import { useOnboardingWizardStore } from "../../store";
import { COUNTRIES } from "../countries";
import { TEAM_INTAKE_FORM_URL } from "./helpers";

export function useSubscriptionStep() {
  const [billing, setBilling] = useState<"monthly" | "yearly">("monthly");
  const [countryIdx, setCountryIdx] = useState(0);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);

  const country = COUNTRIES[countryIdx];
  const isYearly = billing === "yearly";

  function handlePlanSelect(planKey: string) {
    if (planKey === "ENTERPRISE") {
      window.open(TEAM_INTAKE_FORM_URL, "_blank");
      return;
    }
    nextStep();
  }

  return {
    billing,
    setBilling,
    countryIdx,
    setCountryIdx,
    country,
    isYearly,
    handlePlanSelect,
  };
}
