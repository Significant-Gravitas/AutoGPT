import { DescriptionFieldProps } from "@rjsf/utils";
import { RichDescription } from "@rjsf/core";
import { InfoIcon } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

export default function DescriptionField(props: DescriptionFieldProps) {
  const { id, description, registry, uiSchema } = props;
  if (!description) {
    return null;
  }

  return (
    <div id={id} className="0 inline w-fit">
      <Tooltip>
        <TooltipTrigger asChild>
          <InfoIcon size={16} className="cursor-pointer" />
        </TooltipTrigger>
        <TooltipContent>
          <RichDescription
            description={description}
            registry={registry}
            uiSchema={uiSchema}
          />
        </TooltipContent>
      </Tooltip>
    </div>
  );
}
