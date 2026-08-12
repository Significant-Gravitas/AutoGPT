import { DescriptionFieldProps } from "@rjsf/utils";
import { RichDescription } from "@rjsf/core";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export default function DescriptionField(props: DescriptionFieldProps) {
  const { id, description, registry, uiSchema } = props;
  if (!description) {
    return null;
  }

  return (
    <div id={id} className="0 inline w-fit">
      <Tooltip>
        <TooltipTrigger asChild>
          <Icon
            icon={InformationCircleIcon}
            size={16}
            className="cursor-pointer"
          />
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
