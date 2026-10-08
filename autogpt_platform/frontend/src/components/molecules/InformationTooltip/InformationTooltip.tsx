import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import ReactMarkdown from "react-markdown";

type Props = {
  description?: string;
  iconSize?: number;
};

export function InformationTooltip({ description, iconSize = 24 }: Props) {
  if (!description) return null;

  return (
    <TooltipProvider delayDuration={400}>
      <Tooltip>
        {/* A native button, because an SVG cannot take focus: bound straight
            to the icon, the tooltip opened on hover only and a keyboard user
            had no way to read what it holds. Inside a <label> a click would
            be forwarded to the labelled control, which is not what pressing
            an info mark asks for. */}
        <TooltipTrigger
          render={
            <button
              type="button"
              aria-label="More information"
              onClick={(event) => event.preventDefault()}
              className="inline-flex flex-none items-center justify-center rounded-full p-1 text-current hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-hidden"
            >
              <Icon icon={InformationCircleIcon} size={iconSize} aria-hidden />
            </button>
          }
        />
        <TooltipContent className="block text-start font-normal">
          <ReactMarkdown
            components={{
              a: ({ node: _, ...props }) => (
                <a target="_blank" className="underline" {...props} />
              ),
            }}
          >
            {description}
          </ReactMarkdown>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
