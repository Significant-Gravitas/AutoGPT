import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { RJSFSchema } from "@rjsf/utils";
import { uiSchema } from "../../../FlowEditor/nodes/uiSchema";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { Button } from "@/components/atoms/Button/Button";
import { ClockIcon, PlayIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { FormRenderer } from "@/components/renderers/input-renderer/FormRenderer";
import { useRunInputDialog } from "./useRunInputDialog";
import { CronSchedulerDialog } from "../CronSchedulerDialog/CronSchedulerDialog";

export const RunInputDialog = ({
  isOpen,
  setIsOpen,
  purpose,
}: {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  purpose: "run" | "schedule";
}) => {
  const hasInputs = useGraphStore((state) => state.hasInputs);
  const hasCredentials = useGraphStore((state) => state.hasCredentials);
  const inputSchema = useGraphStore((state) => state.inputSchema);
  const credentialsSchema = useGraphStore(
    (state) => state.credentialsInputSchema,
  );

  const {
    credentialsUiSchema,
    handleManualRun,
    handleInputChange,
    openCronSchedulerDialog,
    setOpenCronSchedulerDialog,
    inputValues,
    credentialValues,
    handleCredentialChange,
    isExecutingGraph,
  } = useRunInputDialog({ setIsOpen });

  return (
    <>
      <Dialog
        title={purpose === "run" ? "Run Agent" : "Schedule Run"}
        controlled={{
          isOpen,
          set: setIsOpen,
        }}
        styling={{ maxWidth: "600px", minWidth: "600px" }}
      >
        <Dialog.Content>
          <div className="space-y-6 p-1">
            {/* Credentials Section */}
            {hasCredentials() && (
              <div>
                <div className="mb-4">
                  <Text variant="h4" className="text-gray-900">
                    Credentials
                  </Text>
                </div>
                <div className="px-2">
                  <FormRenderer
                    jsonSchema={credentialsSchema as RJSFSchema}
                    handleChange={(v) => handleCredentialChange(v.formData)}
                    uiSchema={credentialsUiSchema}
                    initialValues={{}}
                    formContext={{
                      showHandles: false,
                      size: "large",
                    }}
                  />
                </div>
              </div>
            )}

            {/* Inputs Section */}
            {hasInputs() && (
              <div>
                <div className="mb-4">
                  <Text variant="h4" className="text-gray-900">
                    Inputs
                  </Text>
                </div>
                <div className="px-2">
                  <FormRenderer
                    jsonSchema={inputSchema as RJSFSchema}
                    handleChange={(v) => handleInputChange(v.formData)}
                    uiSchema={uiSchema}
                    initialValues={{}}
                    formContext={{
                      showHandles: false,
                      size: "large",
                    }}
                  />
                </div>
              </div>
            )}

            {/* Action Button */}
            <div className="flex justify-end pt-2">
              {purpose === "run" && (
                <Button
                  variant="primary"
                  size="large"
                  className="group h-fit min-w-0 gap-2"
                  onClick={handleManualRun}
                  loading={isExecutingGraph}
                >
                  <PlayIcon className="size-5 transition-transform group-hover:scale-110" />
                  <span className="font-semibold">Manual Run</span>
                </Button>
              )}
              {purpose === "schedule" && (
                <Button
                  variant="primary"
                  size="large"
                  className="group h-fit min-w-0 gap-2"
                  onClick={() => setOpenCronSchedulerDialog(true)}
                >
                  <ClockIcon className="size-5 transition-transform group-hover:scale-110" />
                  <span className="font-semibold">Schedule Run</span>
                </Button>
              )}
            </div>
          </div>
        </Dialog.Content>
      </Dialog>
      <CronSchedulerDialog
        open={openCronSchedulerDialog}
        setOpen={setOpenCronSchedulerDialog}
        inputs={inputValues}
        credentials={credentialValues}
      />
    </>
  );
};
