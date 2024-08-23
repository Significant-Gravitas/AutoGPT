import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import ReactMarkdown from "react-markdown";

const SchemaTooltip: React.FC<{ description?: string }> = ({ description }) => {
  if (!description) return null;

  return (
    <TooltipProvider delayDuration={400}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="rounded-full p-1 hover:bg-gray-300" size={24} />
        </TooltipTrigger>
        <TooltipContent className="tooltip-content max-w-xs">
          <ReactMarkdown
            components={{
              a: ({ node, ...props }) => (
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
};

export default SchemaTooltip;
