import { RoleStep } from "@/features/onboarding/components/RoleStep";
import { useOnboardingFlow } from "@/features/onboarding/useOnboardingFlow";

export default function OnboardingScreen() {
  const flow = useOnboardingFlow();

  if (flow.step === "role") {
    return (
      <RoleStep
        role={flow.role}
        otherRole={flow.otherRole}
        onPick={flow.pickRole}
        onChangeOther={flow.setOtherRole}
        onNext={flow.goToWork}
      />
    );
  }
  return null;
}
