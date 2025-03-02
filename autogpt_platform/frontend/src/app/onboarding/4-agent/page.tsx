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

// Will try to use the two first available agents
const fallbackAgents = [
  // Prod
  {
    username: "autogpt",
    slug: "domain-drop-catcher",
  },
  {
    username: "autogpt",
    slug: "personalized-morning-coffee-newsletter",
  },
  // Dev
  {
    username: "autogpt",
    slug: "business-owner-finder",
  },
  {
    username: "autogpt",
    slug: "financial-analysis-agent-your-personalized-financial-insights-tool",
  },
  // Temp
  //todo kcze remove
  {
    username: "bright-eagle-40613",
    slug: "subtractor",
  },
  {
    username: "bright-eagle-40613",
    slug: "power",
  },
];

function isEmptyOrWhitespace(str: string | undefined | null): boolean {
  return !str || str.trim().length === 0;
}

export default function Page() {
  const { state, setState } = useOnboarding(4);
  const [agents, setAgents] = useState<StoreAgentDetails[]>([]);
  const api = useBackendAPI();

  useEffect(() => {
    Promise.allSettled(
      fallbackAgents
        .map((agent) => {
          try {
            return api.getStoreAgent(agent.username, agent.slug);
          } catch {
            //todo kcze: supressing errors for now as some are expected
            // until proper agent choosing logic is implemented
            console.debug(
              `Failed to fetch agent ${agent.username}/${agent.slug}`,
            );
            return null;
          }
        })
        .filter((agent) => agent !== null),
    ).then((agents) => {
      // Set only two fulfilled agents
      setAgents(
        agents
          .filter((agent) => agent.status === "fulfilled")
          .map((agent) => agent.value)
          .slice(0, 2),
      );
    });
  }, [api, setAgents]);

  useEffect(() => {
    // Deselect agent if it's not in the list of agents
    if (
      state?.selectedAgentSlug &&
      !fallbackAgents.some((agent) => agent.slug === state.selectedAgentSlug)
    ) {
      setState({
        selectedAgentSlug: undefined,
        selectedAgentCreator: undefined,
        agentInput: undefined,
      });
    }
  }, [state?.selectedAgentSlug, setState]);

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
          {...(agents[0] || {})}
          selected={
            agents[0] !== undefined
              ? state?.selectedAgentSlug == agents[0]?.slug
              : false
          }
          onClick={() =>
            setState({
              selectedAgentSlug: agents[0].slug,
              selectedAgentCreator: agents[0].creator,
            })
          }
        />
        <OnboardingAgentCard
          {...(agents[1] || {})}
          selected={
            agents[1] !== undefined
              ? state?.selectedAgentSlug == agents[1]?.slug
              : false
          }
          onClick={() =>
            setState({
              selectedAgentSlug: agents[1].slug,
              selectedAgentCreator: agents[1].creator,
            })
          }
        />
      </div>

      <OnboardingFooter>
        <OnboardingButton
          href="/onboarding/5-run"
          disabled={isEmptyOrWhitespace(state?.selectedAgentSlug)}
        >
          Next
        </OnboardingButton>
      </OnboardingFooter>
    </OnboardingStep>
  );
}
