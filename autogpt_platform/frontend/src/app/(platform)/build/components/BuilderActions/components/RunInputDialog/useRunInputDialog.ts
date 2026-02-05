import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";

import {
  ApiError,
  CredentialsMetaInput,
  GraphExecutionMeta,
} from "@/lib/autogpt-server-api";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useCallback, useMemo, useState } from "react";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useReactFlow } from "@xyflow/react";
import type { CredentialField } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";

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
            const errorData = error.response?.detail || {
              node_errors: {},
              message: undefined,
            };
            const nodeErrors = errorData.node_errors || {};

            if (Object.keys(nodeErrors).length > 0) {
              Object.entries(nodeErrors).forEach(
                ([nodeId, nodeErrorsForNode]) => {
                  useNodeStore
                    .getState()
                    .updateNodeErrors(
                      nodeId,
                      nodeErrorsForNode as { [key: string]: string },
                    );
                },
              );
            } else {
              useNodeStore.getState().nodes.forEach((node) => {
                useNodeStore.getState().updateNodeErrors(node.id, {});
              });
            }

            toast({
              title: errorData?.message || "Graph validation failed",
              description:
                "Please fix the validation errors on the highlighted nodes and try again.",
              variant: "destructive",
            });
            setIsOpen(false);

            const firstBackendId = Object.keys(nodeErrors)[0];

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

  // Convert credentials schema to credential fields array for CredentialsGroupedView
  const credentialFields: CredentialField[] = useMemo(() => {
    if (!credentialsSchema?.properties) return [];
    return Object.entries(credentialsSchema.properties);
  }, [credentialsSchema]);

  // Get required credentials as a Set
  const requiredCredentials = useMemo(() => {
    return new Set<string>(credentialsSchema?.required || []);
  }, [credentialsSchema]);

  // Handler for individual credential changes
  const handleCredentialFieldChange = useCallback(
    (key: string, value?: CredentialsMetaInput) => {
      setCredentialValues((prev) => {
        if (value) {
          return { ...prev, [key]: value };
        } else {
          const next = { ...prev };
          delete next[key];
          return next;
        }
      });
    },
    [],
  );

  const handleManualRun = async () => {
    // Filter out incomplete credentials (those without a valid id)
    // RJSF auto-populates const values (provider, type) but not id field
    const validCredentials = Object.fromEntries(
      Object.entries(credentialValues).filter(([_, cred]) => cred && cred.id),
    );

    useNodeStore.getState().clearAllNodeExecutionResults();
    useNodeStore.getState().cleanNodesStatuses();

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
    credentialFields,
    requiredCredentials,
    inputValues,
    credentialValues,
    isExecutingGraph,
    handleInputChange,
    handleCredentialChange,
    handleCredentialFieldChange,
    handleManualRun,
    openCronSchedulerDialog,
    setOpenCronSchedulerDialog,
  };
};
