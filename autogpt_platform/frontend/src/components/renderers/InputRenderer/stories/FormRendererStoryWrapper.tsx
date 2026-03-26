import { useState } from "react";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { FormRenderer } from "../FormRenderer";
import type { RJSFSchema } from "@rjsf/utils";
import type { ExtendedFormContextType } from "../types";

const defaultFormContext: ExtendedFormContextType = {
  nodeId: "story-node",
  showHandles: false,
  size: "medium",
  showOptionalToggle: true,
};

export function FormRendererStory({
  jsonSchema,
  initialValues,
}: {
  jsonSchema: RJSFSchema;
  initialValues?: Record<string, unknown>;
}) {
  const [formData, setFormData] = useState(initialValues ?? {});

  return (
    <div className="w-[400px]">
      <FormRenderer
        jsonSchema={jsonSchema}
        handleChange={(e) => setFormData(e.formData ?? {})}
        uiSchema={{}}
        initialValues={formData}
        formContext={defaultFormContext}
      />
    </div>
  );
}

export function storyDecorator(Story: React.ComponentType) {
  return (
    <TooltipProvider>
      <Story />
    </TooltipProvider>
  );
}
