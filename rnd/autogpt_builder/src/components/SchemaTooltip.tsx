import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";
import { Info } from "lucide-react";
import ReactMarkdown from "react-markdown";

const SchemaTooltip: React.FC<{ schema: BlockIOSubSchema }> = ({ schema }) => {
  if (!schema.description) return null;

  return (
    <TooltipProvider delayDuration={400}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="p-1 rounded-full hover:bg-gray-300" size={24} />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs tooltip-content">
          <ReactMarkdown
            components={{
              a: ({ node, ...props }) => (
                <a className="text-blue-400 underline" {...props} />
              ),
            }}
          >
            {schema.description}
          </ReactMarkdown>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default SchemaTooltip;
