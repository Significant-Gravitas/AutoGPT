"use client";

import OnboardingButton from "../components/OnboardingButton";
import { OnboardingHeader, OnboardingStep } from "../components/OnboardingStep";
import { OnboardingText } from "../components/OnboardingText";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/__legacy__/ui/card";
import type { InputValues } from "./types";
import { Play } from "lucide-react";
import { RunAgentInputs } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/RunAgentInputs/RunAgentInputs";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { isRunDisabled } from "./helpers";
import { useOnboardingRunStep } from "./useOnboardingRunStep";
import { RunAgentHint } from "./components/RunAgentHint";
import { SelectedAgentCard } from "./components/SelectedAgentCard";
import { AgentOnboardingCredentials } from "./components/AgentOnboardingCredentials/AgentOnboardingCredentials";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";

export default function Page() {
  const {
    ready,
    error,
    showInput,
    agent,
    onboarding,
    storeAgent,
    runningAgent,
    credentialsValid,
    credentialsLoaded,
    handleSetAgentInput,
    handleRunAgent,
    handleNewRun,
    handleCredentialsChange,
    handleCredentialsValidationChange,
    handleCredentialsLoadingChange,
  } = useOnboardingRunStep();

  if (!ready) {
    return (
      <div className="flex flex-col gap-4">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
      </div>
    );
  }

  if (error) {
    return <ErrorCard responseError={error} />;
  }

  return (
    <OnboardingStep dotted>
      <OnboardingHeader backHref={"/onboarding/4-agent"} transparent />
      <div className="flex min-h-[80vh] items-center justify-center">
        <SelectedAgentCard storeAgent={storeAgent} />
        <div className="w-[481px]" />
        {!showInput ? (
          <RunAgentHint handleNewRun={handleNewRun} />
        ) : (
          <div className="ml-[104px] w-[481px] pl-5">
            <div className="flex flex-col">
              <OnboardingText variant="header">
                Provide details for your agent
              </OnboardingText>
              <span className="mt-9 text-base font-normal leading-normal text-zinc-600">
                Give your agent the details it needs to work—just enter <br />
                the key information and get started.
              </span>
              <span className="mt-4 text-base font-normal leading-normal text-zinc-600">
                When you&apos;re done, click <b>Run Agent</b>.
              </span>

              <Card className="agpt-box mt-4">
                <CardHeader>
                  <CardTitle className="font-poppins text-lg">Input</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col gap-4">
                  {Object.entries(agent?.input_schema.properties || {}).map(
                    ([key, inputSubSchema]) => (
                      <div key={key} className="flex flex-col space-y-2">
                        <label className="flex items-center gap-1 text-sm font-medium">
                          {inputSubSchema.title || key}
                          <InformationTooltip
                            description={inputSubSchema.description}
                          />
                        </label>
                        <RunAgentInputs
                          schema={inputSubSchema}
                          value={onboarding.state?.agentInput?.[key]}
                          placeholder={inputSubSchema.description}
                          onChange={(value) => handleSetAgentInput(key, value)}
                        />
                      </div>
                    ),
                  )}
                  <AgentOnboardingCredentials
                    agent={agent}
                    siblingInputs={
                      (onboarding.state?.agentInput as Record<string, any>) ||
                      undefined
                    }
                    onCredentialsChange={handleCredentialsChange}
                    onValidationChange={handleCredentialsValidationChange}
                    onLoadingChange={handleCredentialsLoadingChange}
                  />
                </CardContent>
              </Card>
              <OnboardingButton
                variant="violet"
                className="mt-8 w-[136px]"
                loading={runningAgent}
                disabled={isRunDisabled({
                  agent,
                  isRunning: runningAgent,
                  agentInputs:
                    (onboarding.state?.agentInput as unknown as InputValues) ||
                    null,
                  credentialsValid,
                  credentialsLoaded,
                })}
                onClick={handleRunAgent}
                icon={<Play className="mr-2" size={18} />}
              >
                Run agent
              </OnboardingButton>
            </div>
          </div>
        )}
      </div>
    </OnboardingStep>
  );
}
