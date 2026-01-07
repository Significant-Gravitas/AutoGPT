import { RJSFSchema } from "@rjsf/utils";
import React, { useMemo } from "react";
import { uiSchema } from "./uiSchema";
import { useNodeStore } from "../../../stores/nodeStore";
import { BlockUIType } from "../../types";
import { FormRenderer } from "@/components/renderers/input-renderer/FormRenderer";

interface FormCreatorProps {
  jsonSchema: RJSFSchema;
  nodeId: string;
  uiType: BlockUIType;
  showHandles?: boolean;
  className?: string;
}

export const FormCreator: React.FC<FormCreatorProps> = React.memo(
  ({ jsonSchema, nodeId, uiType, showHandles = true, className }) => {
    const updateNodeData = useNodeStore((state) => state.updateNodeData);

    const getHardCodedValues = useNodeStore(
      (state) => state.getHardCodedValues,
    );

    // Subscribe to resolution mode state to get broken inputs
    const resolutionData = useNodeStore((state) =>
      state.nodeResolutionData.get(nodeId),
    );

    // Compute the set of broken input handles (only missing inputs, not type mismatches)
    // Type mismatches still have a valid handle - only the edge is broken, not the handle itself
    const brokenInputs = useMemo(() => {
      if (!resolutionData) return new Set<string>();
      const broken = new Set<string>();
      resolutionData.incompatibilities.missingInputs.forEach((name) =>
        broken.add(name),
      );
      return broken;
    }, [resolutionData]);

    // Compute a map of inputs with type mismatches -> their new type (for highlighting and display)
    const typeMismatchInputs = useMemo(() => {
      if (!resolutionData) return new Map<string, string>();
      const mismatches = new Map<string, string>();
      resolutionData.incompatibilities.inputTypeMismatches.forEach((m) =>
        mismatches.set(m.name, m.newType),
      );
      return mismatches;
    }, [resolutionData]);

    const handleChange = ({ formData }: any) => {
      if ("credentials" in formData && !formData.credentials?.id) {
        delete formData.credentials;
      }

      const updatedValues =
        uiType === BlockUIType.AGENT
          ? {
              ...getHardCodedValues(nodeId),
              inputs: formData,
            }
          : formData;

      updateNodeData(nodeId, { hardcodedValues: updatedValues });
    };

    const hardcodedValues = getHardCodedValues(nodeId);
    const initialValues =
      uiType === BlockUIType.AGENT
        ? (hardcodedValues.inputs ?? {})
        : hardcodedValues;

    return (
      <FormRenderer
        className={className}
        jsonSchema={jsonSchema}
        handleChange={handleChange}
        uiSchema={uiSchema}
        initialValues={initialValues}
        formContext={{
          nodeId: nodeId,
          uiType: uiType,
          showHandles: showHandles,
          size: "small",
          brokenInputs: brokenInputs,
          typeMismatchInputs: typeMismatchInputs,
        }}
      />
    );
  },
);

FormCreator.displayName = "FormCreator";
