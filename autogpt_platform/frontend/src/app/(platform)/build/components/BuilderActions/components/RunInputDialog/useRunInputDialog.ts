import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  CredentialsMetaInput,
  GraphExecutionMeta,
} from "@/lib/autogpt-server-api";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useMemo, useState } from "react";
import { uiSchema } from "../../../FlowEditor/nodes/uiSchema";
import { isCredentialFieldSchema } from "@/components/renderers/InputRenderer/custom/CredentialField/helpers";

export const useRunInputDialog = ({
  setIsOpen,
}: {
  setIsOpen: (isOpen: boolean) => void;
}) => {
  const credentialsSchema = useGraphStore(
    (state) => state.credentialsInputSchema,
  );
  const setIsGraphRunning = useGraphStore((state) => state.setIsGraphRunning);

  const [openCronSchedulerDialog, setOpenCronSchedulerDialog] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [credentialValues, setCredentialValues] = useState<
    Record<string, CredentialsMetaInput>
  >({});
  const [{ flowID, flowVersion }, setQueryStates] = useQueryStates({
    flowExecutionID: parseAsString,
    flowID: parseAsString,
    flowVersion: parseAsInteger,
  });
  const { toast } = useToast();

  const { mutateAsync: executeGraph, isPending: isExecutingGraph } =
    usePostV1ExecuteGraphAgent({
      mutation: {
        onSuccess: (response) => {
          const { id } = response.data as GraphExecutionMeta;
          setQueryStates({
            flowExecutionID: id,
          });
        },
        onError: (error) => {
          // Reset running state on error
          setIsGraphRunning(false);
          toast({
            title: (error.detail as string) ?? "An unexpected error occurred.",
            description: "An unexpected error occurred.",
            variant: "destructive",
          });
        },
      },
    });

  // We are rendering the credentials field differently compared to other fields.
  // In the node, we have the field name as "credential" - so our library catches it and renders it differently.
  // But here we have a different name, something like `Firecrawl credentials`, so here we are telling the library that this field is a credential field type.

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

  const handleManualRun = async () => {
    await executeGraph({
      graphId: flowID ?? "",
      graphVersion: flowVersion || null,
      data: {
        inputs: inputValues,
        credentials_inputs: credentialValues,
        source: "builder",
      },
    });
    // Optimistically set running state immediately for responsive UI
    setIsGraphRunning(true);
    setIsOpen(false);
  };

  const handleInputChange = (inputValues: Record<string, any>) => {
    setInputValues(inputValues);
  };
  const handleCredentialChange = (
    credentialValues: Record<string, CredentialsMetaInput>,
  ) => {
    setCredentialValues(credentialValues);
  };

  return {
    credentialsUiSchema,
    inputValues,
    credentialValues,
    isExecutingGraph,
    handleInputChange,
    handleCredentialChange,
    handleManualRun,
    openCronSchedulerDialog,
    setOpenCronSchedulerDialog,
  };
};
