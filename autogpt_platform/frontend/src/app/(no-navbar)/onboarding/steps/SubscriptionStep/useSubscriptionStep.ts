import { useOnboardingWizardStore } from "../../store";
import { COUNTRIES } from "./countries";
import { PLAN_KEYS, type PlanKey, TEAM_INTAKE_FORM_URL } from "./helpers";

export function useSubscriptionStep() {
  const billing = useOnboardingWizardStore((s) => s.selectedBilling);
  const setSelectedBilling = useOnboardingWizardStore(
    (s) => s.setSelectedBilling,
  );
  const selectedCountryCode = useOnboardingWizardStore(
    (s) => s.selectedCountryCode,
  );
  const setSelectedCountryCode = useOnboardingWizardStore(
    (s) => s.setSelectedCountryCode,
  );
  const setSelectedPlan = useOnboardingWizardStore((s) => s.setSelectedPlan);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);

  const countryIdx = Math.max(
    0,
    COUNTRIES.findIndex((c) => c.countryCode === selectedCountryCode),
  );
  const country = COUNTRIES[countryIdx];
  const isYearly = billing === "yearly";

  function setCountryIdx(idx: number) {
    const next = COUNTRIES[idx];
    if (!next) return;
    setSelectedCountryCode(next.countryCode);
  }

  function handlePlanSelect(planKey: PlanKey) {
    if (planKey === PLAN_KEYS.TEAM) {
      window.open(TEAM_INTAKE_FORM_URL, "_blank", "noopener,noreferrer");
      return;
    }
    setSelectedPlan(planKey);
    nextStep();
  }

  return {
    billing,
    setBilling: setSelectedBilling,
    countryIdx,
    setCountryIdx,
    country,
    isYearly,
    handlePlanSelect,
  };
}
