"use client";
import SmartImage from "@/components/agptui/SmartImage";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import OnboardingButton from "@/components/onboarding/OnboardingButton";
import {
  OnboardingHeader,
  OnboardingStep,
} from "@/components/onboarding/OnboardingStep";
import { OnboardingText } from "@/components/onboarding/OnboardingText";
import StarRating from "@/components/onboarding/StarRating";
import SchemaTooltip from "@/components/SchemaTooltip";
import { TypeBasedInput } from "@/components/type-based-input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/components/ui/use-toast";
import { GraphMeta, StoreAgentDetails } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { cn } from "@/lib/utils";
import { Play } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

export default function Page() {
  const { state, updateState, setStep } = useOnboarding(
    undefined,
    "AGENT_CHOICE",
  );
  const [showInput, setShowInput] = useState(false);
  const [agent, setAgent] = useState<GraphMeta | null>(null);
  const [storeAgent, setStoreAgent] = useState<StoreAgentDetails | null>(null);
  const [runningAgent, setRunningAgent] = useState(false);
  const { toast } = useToast();
  const router = useRouter();
  const api = useBackendAPI();

  useEffect(() => {
    setStep(5);
  }, [setStep]);

  useEffect(() => {
    if (!state?.selectedStoreListingVersionId) {
      return;
    }
    api
      .getStoreAgentByVersionId(state?.selectedStoreListingVersionId)
      .then((storeAgent) => {
        setStoreAgent(storeAgent);
      });
    api
      .getAgentMetaByStoreListingVersionId(state?.selectedStoreListingVersionId)
      .then((agent) => {
        setAgent(agent);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const update: { [key: string]: any } = {};
        // Set default values from schema
        Object.entries(agent.input_schema?.properties || {}).forEach(
          ([key, value]) => {
            // Skip if already set
            if (state.agentInput && state.agentInput[key]) {
              update[key] = state.agentInput[key];
              return;
            }
            update[key] = value.type !== "null" ? value.default || "" : "";
          },
        );
        updateState({
          agentInput: update,
        });
      });
  }, [api, setAgent, updateState, state?.selectedStoreListingVersionId]);

  const setAgentInput = useCallback(
    (key: string, value: string) => {
      updateState({
        agentInput: {
          ...state?.agentInput,
          [key]: value,
        },
      });
    },
    [state?.agentInput, updateState],
  );

  const runAgent = useCallback(async () => {
    if (!agent) {
      return;
    }
    setRunningAgent(true);
    try {
      const libraryAgent = await api.addMarketplaceAgentToLibrary(
        storeAgent?.store_listing_version_id || "",
      );
      const { graph_exec_id } = await api.executeGraph(
        libraryAgent.graph_id,
        libraryAgent.graph_version,
        state?.agentInput || {},
      );
      updateState({
        onboardingAgentExecutionId: graph_exec_id,
        agentRuns: (state?.agentRuns || 0) + 1,
      });
      router.push("/onboarding/6-congrats");
    } catch (error) {
      console.error("Error running agent:", error);
      toast({
        title: "Error running agent",
        description:
          "There was an error running your agent. Please try again or try choosing a different agent if it still fails.",
        variant: "destructive",
      });
      setRunningAgent(false);
    }
  }, [api, agent, router, state?.agentInput, storeAgent, updateState, toast]);

  const runYourAgent = (
    <div className="ml-[104px] w-[481px] pl-5">
      <div className="flex flex-col">
        <OnboardingText variant="header">Run your first agent</OnboardingText>
        <span className="mt-9 text-base font-normal leading-normal text-zinc-600">
          A &apos;run&apos; is when your agent starts working on a task
        </span>
        <span className="mt-4 text-base font-normal leading-normal text-zinc-600">
          Click on <b>New Run</b> below to try it out
        </span>

        <div
          onClick={() => {
            setShowInput(true);
            setStep(6);
            updateState({
              completedSteps: [
                ...(state?.completedSteps || []),
                "AGENT_NEW_RUN",
              ],
            });
          }}
          className={cn(
            "mt-16 flex h-[68px] w-[330px] items-center justify-center rounded-xl border-2 border-violet-700 bg-neutral-50",
            "cursor-pointer transition-all duration-200 ease-in-out hover:bg-violet-50",
          )}
        >
          <svg
            width="38"
            height="38"
            viewBox="0 0 32 32"
            xmlns="http://www.w3.org/2000/svg"
          >
            <g stroke="#6d28d9" strokeWidth="1.2" strokeLinecap="round">
              <line x1="16" y1="8" x2="16" y2="24" />
              <line x1="8" y1="16" x2="24" y2="16" />
            </g>
          </svg>
          <span className="ml-3 font-sans text-[19px] font-medium leading-normal text-violet-700">
            New run
          </span>
        </div>
      </div>
    </div>
  );

  return (
    <OnboardingStep dotted>
      <OnboardingHeader backHref={"/onboarding/4-agent"} transparent />
      {/* Agent card */}
      <div className="fixed left-1/4 top-1/2 w-[481px] -translate-x-1/2 -translate-y-1/2">
        <div className="h-[156px] w-[481px] rounded-xl bg-white px-6 pb-5 pt-4">
          <span className="font-sans text-xs font-medium tracking-wide text-zinc-500">
            SELECTED AGENT
          </span>
          {storeAgent ? (
            <div className="mt-4 flex h-20 rounded-lg bg-violet-50 p-2">
              {/* Left image */}
              <SmartImage
                src={storeAgent?.agent_image[0]}
                alt="Agent cover"
                imageContain
                className="w-[350px] rounded-lg"
              />
              {/* Right content */}
              <div className="ml-2 flex flex-1 flex-col">
                <span className="w-[292px] truncate font-sans text-[14px] font-medium leading-normal text-zinc-800">
                  {storeAgent?.agent_name}
                </span>
                <span className="mt-[5px] w-[292px] truncate font-sans text-xs font-normal leading-tight text-zinc-600">
                  by {storeAgent?.creator}
                </span>
                <div className="mt-auto flex w-[292px] justify-between">
                  <span className="mt-1 truncate font-sans text-xs font-normal leading-tight text-zinc-600">
                    {storeAgent?.runs.toLocaleString("en-US")} runs
                  </span>
                  <StarRating
                    className="font-sans text-xs font-normal leading-tight text-zinc-600"
                    starSize={12}
                    rating={storeAgent?.rating || 0}
                  />
                </div>
              </div>
            </div>
          ) : (
            <div className="mt-4 flex h-20 animate-pulse rounded-lg bg-gray-300 p-2" />
          )}
        </div>
      </div>
      <div className="flex min-h-[80vh] items-center justify-center">
        {/* Left side */}
        <div className="w-[481px]" />
        {/* Right side */}
        {!showInput ? (
          runYourAgent
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
                  {Object.entries(agent?.input_schema?.properties || {}).map(
                    ([key, inputSubSchema]) => (
                      <div key={key} className="flex flex-col space-y-2">
                        <label className="flex items-center gap-1 text-sm font-medium">
                          {inputSubSchema.title || key}
                          <SchemaTooltip
                            description={inputSubSchema.description}
                          />
                        </label>
                        <TypeBasedInput
                          schema={inputSubSchema}
                          value={state?.agentInput?.[key]}
                          placeholder={inputSubSchema.description}
                          onChange={(value) => setAgentInput(key, value)}
                        />
                      </div>
                    ),
                  )}
                </CardContent>
              </Card>
              <OnboardingButton
                variant="violet"
                className="mt-8 w-[136px]"
                loading={runningAgent}
                disabled={
                  Object.values(state?.agentInput || {}).some(
                    (value) => String(value).trim() === "",
                  ) ||
                  !agent ||
                  runningAgent
                }
                onClick={runAgent}
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
