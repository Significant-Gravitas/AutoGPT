import React from "react";
import { ArrayFieldTemplateProps } from "@rjsf/utils";
import { ArrayEditor } from "../../components/ArrayEditor/ArrayEditor";

function ArrayFieldTemplate(props: ArrayFieldTemplateProps) {
  const {
    items,
    canAdd,
    onAddClick,
    disabled,
    readonly,
    formContext,
    idSchema,
  } = props;
  const { nodeId } = formContext;

  return (
    <ArrayEditor
      items={items}
      nodeId={nodeId}
      canAdd={canAdd}
      onAddClick={onAddClick}
      disabled={disabled}
      readonly={readonly}
      id={idSchema.$id}
    />
  );
}

export default ArrayFieldTemplate;
