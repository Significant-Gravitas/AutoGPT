import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Info } from "lucide-react";
import ReactMarkdown from "react-markdown";

type Props = {
  description?: string;
  iconSize?: number;
};

export function InformationTooltip({ description, iconSize = 24 }: Props) {
  if (!description) return null;

  return (
    <TooltipProvider delayDuration={400} disableHoverableContent={false}>
      <Tooltip>
        <TooltipTrigger asChild>
          {/* A native button, because an SVG cannot take focus: bound straight
              to the icon, the tooltip opened on hover only and a keyboard user
              had no way to read what it holds. Inside a <label> a click would
              be forwarded to the labelled control, which is not what pressing
              an info mark asks for. */}
          {/* TODO(tokens): no role for slate-50 (#F7F8F9). */}
          <button
            type="button"
            aria-label="More information"
            onClick={(event) => event.preventDefault()}
            data-slot="information-tooltip-trigger"
            className="inline-flex flex-none items-center justify-center rounded-full p-1 text-current hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border-strong"
          >
            <Info
              data-slot="information-tooltip-icon"
              aria-hidden
              size={iconSize}
            />
          </button>
        </TooltipTrigger>
        <TooltipContent data-slot="information-tooltip-popup">
          <ReactMarkdown
            components={{
              a: ({ node: _, ...props }) => (
                // TODO(tokens): no role for blue-400 (#60A5FA).
                <a
                  target="_blank"
                  data-slot="information-tooltip-link"
                  className="pointer-events-auto text-blue-400 underline"
                  {...props}
                />
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
