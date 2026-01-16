import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";

import {
  ApiError,
  CredentialsMetaInput,
  GraphExecutionMeta,
} from "@/lib/autogpt-server-api";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useMemo, useState } from "react";
import { uiSchema } from "../../../FlowEditor/nodes/uiSchema";
import { isCredentialFieldSchema } from "@/components/renderers/InputRenderer/custom/CredentialField/helpers";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useReactFlow } from "@xyflow/react";

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
  const { setViewport } = useReactFlow();

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
          if (error instanceof ApiError && error.isGraphValidationError()) {
            const errorData = error.response?.detail;
            Object.entries(errorData.node_errors).forEach(
              ([nodeId, nodeErrors]) => {
                useNodeStore
                  .getState()
                  .updateNodeErrors(
                    nodeId,
                    nodeErrors as { [key: string]: string },
                  );
              },
            );
            toast({
              title: errorData?.message || "Graph validation failed",
              description:
                "Please fix the validation errors on the highlighted nodes and try again.",
              variant: "destructive",
            });
            setIsOpen(false);

            const firstBackendId = Object.keys(errorData.node_errors)[0];

            if (firstBackendId) {
              const firstErrorNode = useNodeStore
                .getState()
                .nodes.find(
                  (n) =>
                    n.data.metadata?.backend_id === firstBackendId ||
                    n.id === firstBackendId,
                );

              if (firstErrorNode) {
                setTimeout(() => {
                  setViewport(
                    {
                      x:
                        -firstErrorNode.position.x * 0.8 +
                        window.innerWidth / 2 -
                        150,
                      y: -firstErrorNode.position.y * 0.8 + 50,
                      zoom: 0.8,
                    },
                    { duration: 500 },
                  );
                }, 50);
              }
            }
          } else {
            toast({
              title: "Error running graph",
              description:
                (error as Error).message || "An unexpected error occurred.",
              variant: "destructive",
            });
            setIsOpen(false);
          }
          setIsGraphRunning(false);
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
            "ui:field": "custom/credential_field",
          };
        }
      });
    }

    return dynamicUiSchema;
  }, [credentialsSchema]);

  const handleManualRun = async () => {
    // Filter out incomplete credentials (those without a valid id)
    // RJSF auto-populates const values (provider, type) but not id field
    const validCredentials = Object.fromEntries(
      Object.entries(credentialValues).filter(([_, cred]) => cred && cred.id),
    );

    await executeGraph({
      graphId: flowID ?? "",
      graphVersion: flowVersion || null,
      data: {
        inputs: inputValues,
        credentials_inputs: validCredentials,
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
