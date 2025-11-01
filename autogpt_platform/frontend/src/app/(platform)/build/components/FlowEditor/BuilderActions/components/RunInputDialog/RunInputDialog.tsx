import { Dialog } from "@/components/molecules/Dialog/Dialog";
import Form from "@rjsf/core";
import { RJSFSchema } from "@rjsf/utils";
import { widgets } from "@/components/form-renderer/widgets";
import { fields } from "@/components/form-renderer/fields";
import { templates } from "@/components/form-renderer/templates";
import { uiSchema } from "../../../nodes/uiSchema";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useMemo } from "react";
import { preprocessInputSchema } from "../../../../../../../../components/form-renderer/utils/input-schema-pre-processor";
import validator from "@rjsf/validator-ajv8";

export const RunInputDialog = ({
  isOpen,
  setIsOpen,
}: {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
}) => {
  const inputSchema = useGraphStore((state) => state.inputSchema);
  const credentialsSchema = useGraphStore(
    (state) => state.credentialsInputSchema,
  );
  const preprocessedInputSchema = useMemo(() => {
    return preprocessInputSchema(inputSchema as RJSFSchema);
  }, [inputSchema]);

  const preprocessedCredentialsSchema = useMemo(() => {
    return preprocessInputSchema(credentialsSchema as RJSFSchema);
  }, [credentialsSchema]);

  return (
    <Dialog
      title="Run Agent"
      controlled={{
        isOpen,
        set: setIsOpen,
      }}
    >
      <Dialog.Content>
        <Form
          schema={preprocessedInputSchema}
          fields={fields}
          templates={templates}
          widgets={widgets}
          uiSchema={uiSchema}
          validator={validator}
          formContext={{
            showHandles: false,
            size: "large",
          }}
        />
      </Dialog.Content>
    </Dialog>
  );
};
