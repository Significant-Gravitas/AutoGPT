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
};

export function InformationTooltip({ description }: Props) {
  if (!description) return null;

  return (
    <TooltipProvider delayDuration={400}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="rounded-full p-1 hover:bg-slate-50" size={24} />
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
