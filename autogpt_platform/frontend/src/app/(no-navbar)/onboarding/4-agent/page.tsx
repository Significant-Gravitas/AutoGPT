"use client";
import OnboardingButton from "../components/OnboardingButton";
import {
  OnboardingFooter,
  OnboardingHeader,
  OnboardingStep,
} from "../components/OnboardingStep";
import { OnboardingText } from "../components/OnboardingText";
import OnboardingAgentCard from "../components/OnboardingAgentCard";
import { useEffect, useState } from "react";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { StoreAgentDetails } from "@/lib/autogpt-server-api";
import { finishOnboarding } from "../6-congrats/actions";
import { isEmptyOrWhitespace } from "@/lib/utils";
import { useOnboarding } from "../../../../providers/onboarding/onboarding-provider";

export default function Page() {
  const { state, updateState } = useOnboarding(4, "INTEGRATIONS");
  const [agents, setAgents] = useState<StoreAgentDetails[]>([]);
  const api = useBackendAPI();

  useEffect(() => {
    api.getOnboardingAgents().then((agents) => {
      if (agents.length < 2) {
        finishOnboarding();
      }
      setAgents(agents);
    });
  }, [api, setAgents]);

  useEffect(() => {
    // Deselect agent if it's not in the list of agents
    if (
      state?.selectedStoreListingVersionId &&
      agents.length > 0 &&
      !agents.some(
        (agent) =>
          agent.store_listing_version_id ===
          state.selectedStoreListingVersionId,
      )
    ) {
      updateState({
        selectedStoreListingVersionId: null,
        agentInput: {},
      });
    }
  }, [state?.selectedStoreListingVersionId, updateState, agents]);

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
          agent={agents[0]}
          selected={
            agents[0] !== undefined
              ? state?.selectedStoreListingVersionId ==
                agents[0]?.store_listing_version_id
              : false
          }
          onClick={() =>
            updateState({
              selectedStoreListingVersionId: agents[0].store_listing_version_id,
              agentInput: {},
            })
          }
        />
        <OnboardingAgentCard
          agent={agents[1]}
          selected={
            agents[1] !== undefined
              ? state?.selectedStoreListingVersionId ==
                agents[1]?.store_listing_version_id
              : false
          }
          onClick={() =>
            updateState({
              selectedStoreListingVersionId: agents[1].store_listing_version_id,
            })
          }
        />
      </div>

      <OnboardingFooter>
        <OnboardingButton
          href="/onboarding/5-run"
          disabled={isEmptyOrWhitespace(state?.selectedStoreListingVersionId)}
        >
          Next
        </OnboardingButton>
      </OnboardingFooter>
    </OnboardingStep>
  );
}
