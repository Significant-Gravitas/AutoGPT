import { RoleStep } from "@/features/onboarding/components/RoleStep";
import { WorkStep } from "@/features/onboarding/components/WorkStep";
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
  if (flow.step === "work") {
    return <WorkStep onSubmit={flow.submitWork} onBackToRole={flow.backToRole} />;
  }
  return null;
}
