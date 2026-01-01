import { OptionalDataControlsTemplateProps } from "@rjsf/utils";
import { PlusCircle } from "lucide-react";

import { IconButton, RemoveButton } from "../standard/buttons";

export default function OptionalDataControlsTemplate(
  props: OptionalDataControlsTemplateProps,
) {
  const { id, registry, label, onAddClick, onRemoveClick } = props;
  if (onAddClick) {
    return (
      <IconButton
        id={id}
        registry={registry}
        className="rjsf-add-optional-data"
        onClick={onAddClick}
        title={label}
        icon={<PlusCircle />}
        size="small"
      />
    );
  } else if (onRemoveClick) {
    return (
      <RemoveButton
        id={id}
        registry={registry}
        className="rjsf-remove-optional-data"
        onClick={onRemoveClick}
        title={label}
        size="small"
      />
    );
  }
  return <em id={id}>{label}</em>;
}
