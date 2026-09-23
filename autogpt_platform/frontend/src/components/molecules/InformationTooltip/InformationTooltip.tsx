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
    <TooltipProvider delayDuration={400}>
      <Tooltip>
        <TooltipTrigger asChild>
          {/* A native button, because an SVG cannot take focus: bound straight
              to the icon, the tooltip opened on hover only and a keyboard user
              had no way to read what it holds. Inside a <label> a click would
              be forwarded to the labelled control, which is not what pressing
              an info mark asks for. */}
          <button
            type="button"
            aria-label="More information"
            onClick={(event) => event.preventDefault()}
            className="inline-flex flex-none items-center justify-center rounded-full p-1 text-current hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
          >
            <Info aria-hidden size={iconSize} />
          </button>
        </TooltipTrigger>
        <TooltipContent>
          <ReactMarkdown
            components={{
              a: ({ node: _, ...props }) => (
                <a
                  target="_blank"
                  className="text-blue-400 underline"
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
