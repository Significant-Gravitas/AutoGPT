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

/** The `DescriptionField` is the template to use to render the description of a field
 *
 * @param props - The `DescriptionFieldProps` for this component
 */
export default function DescriptionField<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({ id, description, registry, uiSchema }: DescriptionFieldProps<T, S, F>) {
  if (!description) {
    return null;
  }

  return (
    <div id={id} className="inline w-fit bg-red-500">
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
