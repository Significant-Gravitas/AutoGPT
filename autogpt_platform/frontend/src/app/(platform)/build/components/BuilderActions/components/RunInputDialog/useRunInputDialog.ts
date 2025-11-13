import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  CredentialsMetaInput,
  GraphExecutionMeta,
} from "@/lib/autogpt-server-api";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { uiSchema } from "../../../FlowEditor/nodes/uiSchema";
import { isCredentialFieldSchema } from "@/components/renderers/input-renderer/fields/CredentialField/helpers";

export const useRunInputDialog = ({
  setIsOpen,
}: {
  setIsOpen: (isOpen: boolean) => void;
}) => {
  const credentialsSchema = useGraphStore(
    (state) => state.credentialsInputSchema,
  );

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
  const setIsGraphRunning = useGraphStore(
    useShallow((state) => state.setIsGraphRunning),
  );
  const { toast } = useToast();

  const { mutateAsync: executeGraph, isPending: isExecutingGraph } =
    usePostV1ExecuteGraphAgent({
      mutation: {
        onSuccess: (response) => {
          const { id } = response.data as GraphExecutionMeta;
          setQueryStates({
            flowExecutionID: id,
          });
          setIsGraphRunning(false);
        },
        onError: (error) => {
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

  const handleManualRun = () => {
    setIsOpen(false);
    setIsGraphRunning(true);
    executeGraph({
      graphId: flowID ?? "",
      graphVersion: flowVersion || null,
      data: { inputs: inputValues, credentials_inputs: credentialValues },
    });
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
