import {
  DescriptionFieldProps,
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { RichDescription } from "@rjsf/core";
import { InfoIcon } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

export default function DescriptionField<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({ id, description, registry, uiSchema }: DescriptionFieldProps<T, S, F>) {
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
