import { ArrayFieldTemplateItemType, RJSFSchema } from "@rjsf/utils";
import { generateHandleId, HandleIdType } from "../../handlers/helpers";
import { ArrayEditorContext } from "./ArrayEditorContext";
import { Button } from "@/components/atoms/Button/Button";
import { PlusIcon, XIcon } from "@phosphor-icons/react";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";

export interface ArrayEditorProps {
  items?: ArrayFieldTemplateItemType<any, RJSFSchema, any>[];
  nodeId: string;
  canAdd: boolean | undefined;
  onAddClick?: () => void;
  disabled: boolean | undefined;
  readonly: boolean | undefined;
  id: string;
}

export const ArrayEditor = ({
  items,
  nodeId,
  canAdd,
  onAddClick,
  disabled,
  readonly,
  id,
}: ArrayEditorProps) => {
  const { isInputConnected } = useEdgeStore();

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <div className="flex-1">
          {items?.map((element) => {
            const fieldKey = generateHandleId(
              id,
              [element.index.toString()],
              HandleIdType.ARRAY,
            );
            const isConnected = isInputConnected(nodeId, fieldKey);
            return (
              <div
                key={element.key}
                className="-ml-2 flex max-w-[400px] items-center gap-2"
              >
                <ArrayEditorContext.Provider
                  value={{
                    isArrayItem: true,
                    fieldKey,
                    isConnected,
                  }}
                >
                  {element.children}
                </ArrayEditorContext.Provider>

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
                      <XIcon className="h-4 w-4" />
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
          <PlusIcon className="mr-2 h-4 w-4" />
          Add Item
        </Button>
      )}
    </div>
  );
};
