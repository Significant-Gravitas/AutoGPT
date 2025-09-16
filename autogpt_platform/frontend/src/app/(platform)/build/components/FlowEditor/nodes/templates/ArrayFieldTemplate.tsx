import React from "react";
import { ArrayFieldTemplateProps } from "@rjsf/utils";
import { Plus, X } from "lucide-react";
import { Button } from "@/components/atoms/Button/Button";

function ArrayFieldTemplate(props: ArrayFieldTemplateProps) {
  const { items, canAdd, onAddClick, disabled, readonly } = props;

  return (
    <div className="space-y-2">
      <div>
        {items.map((element) => (
          <div key={element.key} className="-ml-2 flex items-center gap-2">
            {element.children}

            {element.hasRemove && !readonly && !disabled && (
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
        ))}
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
