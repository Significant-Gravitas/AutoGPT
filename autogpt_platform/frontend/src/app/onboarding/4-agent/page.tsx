"use client";
import OnboardingButton from "@/components/onboarding/OnboardingButton";
import { useOnboarding } from "../layout";
import {
  OnboardingFooter,
  OnboardingHeader,
  OnboardingStep,
} from "@/components/onboarding/OnboardingStep";
import { OnboardingText } from "@/components/onboarding/OnboardingText";
import OnboardingAgentCard from "@/components/onboarding/OnboardingAgentCard";
import { useEffect, useState } from "react";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { StoreAgentDetails } from "@/lib/autogpt-server-api";

const storeAgents = [
  {
    username: "",
    agentName: ""
  }
];

function isEmptyOrWhitespace(str: string | undefined | null): boolean {
  return !str || str.trim().length === 0;
}

export default function Page() {
  const { state, setState } = useOnboarding(4);
  const [agents, setAgents] = useState<StoreAgentDetails[]>([]);
  const api = useBackendAPI();

  useEffect(() => {
    api.getStoreAgent(storeAgents[0].username, storeAgents[0].agentName)
      .then((agent) => setAgents([agent]));
  }, [api, setAgents]);

  return (
    <OnboardingStep>
      <OnboardingHeader backHref={"/onboarding/3-services"}>
        <OnboardingText className="mt-4" variant="header" center>
          Choose an agent
        </OnboardingText>
        <OnboardingText className="mt-1" center>
          We think these agents are a good match for you based on your answers
        </OnboardingText>
      </OnboardingHeader>

      <div className="my-12 flex items-center justify-between gap-5">
        <OnboardingAgentCard
          {...agents[0]}
          selected={state?.chosenAgentId == "0"}
          onClick={() => setState({ chosenAgentId: "0" })}
        />
        <OnboardingAgentCard
          {...agents[1]}
          selected={state?.chosenAgentId == "1"}
          onClick={() => setState({ chosenAgentId: "1" })}
        />
      </div>

      <OnboardingFooter>
        <OnboardingButton
          href="/onboarding/5-run"
          disabled={isEmptyOrWhitespace(state?.chosenAgentId)}
        >
          Next
        </OnboardingButton>
      </OnboardingFooter>
    </OnboardingStep>
  );
}
