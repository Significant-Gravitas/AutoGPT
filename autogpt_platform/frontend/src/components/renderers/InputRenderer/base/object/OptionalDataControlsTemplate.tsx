import { getFieldDomId } from "../../field-accessibility";
import { OptionalDataControlsTemplateProps } from "@rjsf/utils";
import { AddCircleIcon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";

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
        icon={<Icon icon={AddCircleIcon} size={24} />}
        size="md"
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
        size="md"
      />
    );
  }
  return <em id={getFieldDomId(id, registry.formContext)}>{label}</em>;
}
