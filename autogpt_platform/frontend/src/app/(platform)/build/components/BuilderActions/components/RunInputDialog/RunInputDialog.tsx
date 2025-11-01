import { Dialog } from "@/components/molecules/Dialog/Dialog";
import Form from "@rjsf/core";
import { RJSFSchema } from "@rjsf/utils";
import { widgets } from "../../../../../../../components/form-renderer/widgets";
import { fields } from "../../../../../../../components/form-renderer/fields";
import { templates } from "../../../../../../../components/form-renderer/templates";
import { uiSchema } from "../../../FlowEditor/nodes/uiSchema";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useMemo } from "react";
import { preprocessInputSchema } from "../../../../../../../components/form-renderer/utils/input-schema-pre-processor";
import validator from "@rjsf/validator-ajv8";
import { Button } from "@/components/atoms/Button/Button";
import { PlayIcon, LightningIcon, KeyIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { isCredentialFieldSchema } from "@/components/form-renderer/fields/CredentialField/helpers";

export const RunInputDialog = ({
  isOpen,
  setIsOpen,
}: {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
}) => {
  const hasInputs = useGraphStore((state) => state.hasInputs);
  const hasCredentials = useGraphStore((state) => state.hasCredentials);
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

  const credentialsUiSchema = useMemo(() => {
    const dynamicUiSchema: any = { ...uiSchema };

    if (credentialsSchema?.properties) {
      Object.keys(credentialsSchema.properties).forEach((fieldName) => {
        const fieldSchema = credentialsSchema.properties[fieldName];
        if (isCredentialFieldSchema(fieldSchema)) {
          dynamicUiSchema[fieldName] = {
            ...dynamicUiSchema[fieldName],
            "ui:field": "credentials",
          };
        }
      });
    }

    return dynamicUiSchema;
  }, [credentialsSchema]);

  return (
    <Dialog
      title="Run Agent"
      controlled={{
        isOpen,
        set: setIsOpen,
      }}
      styling={{ maxWidth: "700px" }}
    >
      <Dialog.Content>
        <div className="space-y-6 p-1">
          {/* Credentials Section */}
          <div className="group rounded-xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-6 shadow-sm transition-all">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 shadow-sm">
                <KeyIcon className="size-5 text-white" weight="duotone" />
              </div>
              <div>
                <Text variant="h4" className="text-gray-900">
                  Credentials
                </Text>
                <Text variant="body" className="text-sm text-gray-500">
                  Secure authentication for your agent
                </Text>
              </div>
            </div>
            {hasCredentials() ? (
              <div className="rounded-lg bg-white p-4 shadow-sm ring-1 ring-gray-100">
                <Form
                  schema={preprocessedCredentialsSchema}
                  fields={fields}
                  templates={templates}
                  widgets={widgets}
                  uiSchema={credentialsUiSchema}
                  validator={validator}
                  formContext={{
                    showHandles: false,
                    size: "large",
                  }}
                  className="-mt-8 flex-1"
                />
              </div>
            ) : (
              <div className="flex items-center gap-2 rounded-lg border border-dashed border-gray-300 bg-gray-50/50 px-4 py-3">
                <div className="flex size-8 items-center justify-center rounded-full bg-gray-200">
                  <KeyIcon className="size-4 text-gray-500" />
                </div>
                <Text variant="body" className="text-sm text-gray-600">
                  No credentials required for this agent
                </Text>
              </div>
            )}
          </div>

          {/* Inputs Section */}
          <div className="group rounded-xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-6 shadow-sm transition-all">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-purple-600 shadow-sm">
                <LightningIcon className="size-5 text-white" weight="duotone" />
              </div>
              <div>
                <Text variant="h4" className="text-gray-900">
                  Inputs
                </Text>
                <Text variant="body" className="text-sm text-gray-500">
                  Configure parameters for execution
                </Text>
              </div>
            </div>
            {hasInputs() ? (
              <div className="rounded-lg bg-white p-4 shadow-sm ring-1 ring-gray-100">
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
                  className="-mt-8 flex-1"
                />
              </div>
            ) : (
              <div className="flex items-center gap-2 rounded-lg border border-dashed border-gray-300 bg-gray-50/50 px-4 py-3">
                <div className="flex size-8 items-center justify-center rounded-full bg-gray-200">
                  <LightningIcon className="size-4 text-gray-500" />
                </div>
                <Text variant="body" className="text-sm text-gray-600">
                  No inputs required for this agent
                </Text>
              </div>
            )}
          </div>

          {/* Action Button */}
          <div className="flex justify-end pt-2">
            <Button
              variant="primary"
              size="large"
              className="group h-fit min-w-0 gap-2 border-none bg-gradient-to-r from-blue-600 to-purple-600 px-8 transition-all"
            >
              <PlayIcon className="size-5 transition-transform group-hover:scale-110" />
              <span className="font-semibold">Manual Run</span>
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
};
