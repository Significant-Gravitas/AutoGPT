"use client";
import OnboardingButton from "@/components/onboarding/OnboardingButton";
import {
  OnboardingStep,
  OnboardingHeader,
} from "@/components/onboarding/OnboardingStep";
import { OnboardingText } from "@/components/onboarding/OnboardingText";
import { useOnboarding } from "../layout";
import StarRating from "@/components/onboarding/StarRating";
import { Play } from "lucide-react";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useState } from "react";
import OnboardingAgentInput from "@/components/onboarding/OnboardingAgentInput";
import Image from "next/image";
import { LibraryAgent, StoreAgentDetails } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useRouter } from "next/navigation";

export default function Page() {
  const { state, setState } = useOnboarding(5);
  const [showInput, setShowInput] = useState(false);
  const [storeAgent, setStoreAgent] = useState<StoreAgentDetails | null>(null);
  const [agent, setAgent] = useState<LibraryAgent | null>(null);
  const router = useRouter();
  const api = useBackendAPI();

  useEffect(() => {
    if (!state?.selectedAgentCreator || !state?.selectedAgentSlug) {
      return;
    }
    api
      .getStoreAgent(state?.selectedAgentCreator!, state?.selectedAgentSlug!)
      .then((agent) => {
        setStoreAgent(agent);
        api
          .addMarketplaceAgentToLibrary(agent?.store_listing_version_id!)
          .then((agent) => {
            setAgent(agent);
            const update: { [key: string]: any } = {};
            // Set default values from schema
            Object.entries(agent?.input_schema?.properties || {}).forEach(
              ([key, value]) => {
                // Skip if already set
                if (state?.agentInput && state?.agentInput[key]) {
                  update[key] = state?.agentInput[key];
                  return;
                }
                update[key] = value.type !== "null" ? value.default || "" : "";
              },
            );
            setState({
              agentInput: update,
            });
          });
      });
  }, [
    api,
    setAgent,
    setStoreAgent,
    setState,
    state?.selectedAgentCreator,
    state?.selectedAgentSlug,
  ]);

  const setAgentInput = useCallback(
    (key: string, value: string) => {
      setState({
        ...state,
        agentInput: {
          ...state?.agentInput,
          [key]: value,
        },
      });
    },
    [state, state?.agentInput, setState],
  );

  const runAgent = useCallback(() => {
    if (!agent) {
      return;
    }
    api.executeGraph(agent.agent_id, agent.agent_version, state?.agentInput);
    router.push("/onboarding/6-congrats");
  }, [api, agent, router]);

  const runYourAgent = (
    <div className="ml-[54px] w-[481px] pl-5">
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
            setState({ step: 6 });
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
      <div
        className={cn(
          "flex w-full items-center justify-center",
          showInput ? "mt-[32px]" : "mt-[192px]",
        )}
      >
        {/* Left side */}
        <div className="mr-[52px] w-[481px]">
          <div className="h-[156px] w-[481px] rounded-xl bg-white px-6 pb-5 pt-4">
            <span className="font-sans text-xs font-medium tracking-wide text-zinc-500">
              SELECTED AGENT
            </span>
            <div className="mt-4 flex h-20 rounded-lg bg-violet-50 p-2">
              {/* Left image */}
              <Image
                src={storeAgent?.agent_image[0] || ""}
                alt="Description"
                width={350}
                height={196}
                className="h-full w-auto rounded-lg object-contain"
              />

              {/* Right content */}
              <div className="ml-2 flex flex-1 flex-col">
                <span className="w-[292px] truncate font-sans text-[14px] font-medium leading-normal text-zinc-800">
                  {agent?.name}
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
          </div>
        </div>
        {/* Right side */}
        {!showInput ? (
          runYourAgent
        ) : (
          <div className="ml-[54px] w-[481px] pl-5">
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
              <div className="mt-12 inline-flex w-[492px] flex-col items-start justify-start gap-2 rounded-[20px] border border-zinc-300 bg-white p-6">
                <OnboardingText className="mb-3 font-semibold" variant="header">
                  Input
                </OnboardingText>
                {Object.entries(agent?.input_schema?.properties || {}).map(
                  ([key, value]) => (
                    <OnboardingAgentInput
                      key={key}
                      name={value.title!}
                      description={value.description || ""}
                      placeholder={value.placeholder || ""}
                      value={state?.agentInput?.[key] || ""}
                      onChange={(v) => setAgentInput(key, v)}
                    />
                  ),
                )}
              </div>
              <OnboardingButton
                variant="violet"
                className="mt-8 w-[136px]"
                disabled={
                  Object.values(state?.agentInput || {}).some(
                    (value) => value.trim() === "",
                  ) || !agent
                }
                onClick={runAgent}
              >
                <Play className="" size={18} />
                Run agent
              </OnboardingButton>
            </div>
          </div>
        )}
      </div>
    </OnboardingStep>
  );
}
