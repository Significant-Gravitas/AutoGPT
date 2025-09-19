import React from "react";
import { ArrayFieldTemplateProps } from "@rjsf/utils";
import { Plus, X } from "lucide-react";
import { Button } from "@/components/atoms/Button/Button";
import { generateHandleId, HandleIdType } from "../../handlers/helpers";
import { useEdgeStore } from "../../../store/edgeStore";
import { HandleContext } from "../../handlers/HandleContext";

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
  const { isInputConnected } = useEdgeStore();

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <div className="flex-1">
          {items.map((element) => {
            const fieldKey = generateHandleId(
              idSchema.$id,
              [element.index.toString()],
              HandleIdType.ARRAY,
            );
            const isConnected = isInputConnected(nodeId, fieldKey);
            return (
              <div
                key={element.key}
                className="-ml-2 flex max-w-[400px] items-center gap-2"
              >
                <HandleContext.Provider
                  value={{ isArrayItem: true, fieldKey, isConnected }}
                >
                  {element.children}
                </HandleContext.Provider>

                {element.hasRemove &&
                  !readonly &&
                  !disabled &&
                  !isConnected && (
                    <Button
                      type="button"
                      variant="secondary"
                      className="relative top-5"
                      size="small"
                      onClick={element.onDropIndexClick(element.index)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  )}
              </div>
            );
          })}
        </div>
      </div>

      {canAdd && !readonly && !disabled && (
        <Button
          type="button"
          size="small"
          onClick={onAddClick}
          className="w-full"
        >
          <Plus className="mr-2 h-4 w-4" />
          Add Item
        </Button>
      )}
    </div>
  );
}

export default ArrayFieldTemplate;
