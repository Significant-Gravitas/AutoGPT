import { cn } from "@/lib/utils";
import { OnboardingText } from "../../components/OnboardingText";

type RunAgentHintProps = {
  handleNewRun: () => void;
};

export function RunAgentHint(props: RunAgentHintProps) {
  return (
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
          onClick={props.handleNewRun}
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
}
