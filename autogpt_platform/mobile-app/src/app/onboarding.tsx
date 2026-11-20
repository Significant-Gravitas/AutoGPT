import { PlanNeeded } from "@/features/onboarding/components/PlanNeeded";
import { RoleStep } from "@/features/onboarding/components/RoleStep";
import { SetupStep } from "@/features/onboarding/components/SetupStep";
import { WorkStep } from "@/features/onboarding/components/WorkStep";
import { displayName, resolveRole } from "@/features/onboarding/helpers";
import { useOnboardingFlow } from "@/features/onboarding/useOnboardingFlow";
import { useSession } from "@/lib/auth/nativeSession";

export default function OnboardingScreen() {
  const session = useSession();
  const flow = useOnboardingFlow();

  if (!session) return null;

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
  if (flow.step === "work" || !flow.work) {
    return <WorkStep onSubmit={flow.submitWork} onBackToRole={flow.backToRole} />;
  }
  if (flow.step === "setup") {
    return (
      <SetupStep
        userID={session.user.id}
        userName={displayName(session.user)}
        role={resolveRole(flow.role, flow.otherRole)}
        work={flow.work}
        recordingID={flow.recordingID}
        onPlanNeeded={flow.showPlanNeeded}
      />
    );
  }
  return <PlanNeeded userID={session.user.id} />;
}
