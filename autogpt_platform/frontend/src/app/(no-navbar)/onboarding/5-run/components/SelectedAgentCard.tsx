import { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import StarRating from "../../components/StarRating";
import SmartImage from "@/components/__legacy__/SmartImage";

type Props = {
  storeAgent: StoreAgentDetails | null;
};

export function SelectedAgentCard(props: Props) {
  return (
    <div className="fixed left-1/4 top-1/2 w-[481px] -translate-x-1/2 -translate-y-1/2">
      <div className="h-[156px] w-[481px] rounded-xl bg-white px-6 pb-5 pt-4">
        <span className="font-sans text-xs font-medium tracking-wide text-zinc-500">
          SELECTED AGENT
        </span>
        {props.storeAgent ? (
          <div className="mt-4 flex h-20 rounded-lg bg-violet-50 p-3">
            {/* Left image */}
            <SmartImage
              src={props.storeAgent.agent_image[0]}
              alt="Agent cover"
              className="w-[350px] rounded-lg"
            />
            {/* Right content */}
            <div className="ml-3 flex flex-1 flex-col">
              <div className="mb-2 flex flex-col items-start">
                <span className="data-sentry-unmask w-[292px] truncate font-sans text-[14px] font-medium leading-tight text-zinc-800">
                  {props.storeAgent.agent_name}
                </span>
                <span className="data-sentry-unmask font-norma w-[292px] truncate font-sans text-xs text-zinc-600">
                  by {props.storeAgent.creator}
                </span>
              </div>
              <div className="flex w-[292px] items-center justify-between">
                <span className="truncate font-sans text-xs font-normal leading-tight text-zinc-600">
                  {props.storeAgent.runs.toLocaleString("en-US")} runs
                </span>
                <StarRating
                  className="font-sans text-xs font-normal leading-tight text-zinc-600"
                  starSize={12}
                  rating={props.storeAgent.rating || 0}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-4 flex h-20 animate-pulse rounded-lg bg-gray-300 p-2" />
        )}
      </div>
    </div>
  );
}
