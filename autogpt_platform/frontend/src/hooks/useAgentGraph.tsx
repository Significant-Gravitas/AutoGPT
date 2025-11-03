import { CustomEdge } from "@/app/(platform)/build/components/legacy-builder/CustomEdge/CustomEdge";
import { CustomNode } from "@/app/(platform)/build/components/legacy-builder/CustomNode/CustomNode";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  ApiError,
  Block,
  BlockIOSubSchema,
  BlockUIType,
  CredentialsMetaInput,
  formatEdgeID,
  Graph,
  GraphCreatable,
  GraphExecutionID,
  GraphID,
  GraphMeta,
  LibraryAgent,
  LinkCreatable,
  NodeCreatable,
  NodeExecutionResult,
  SpecialBlockID,
  Node,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { deepEquals, getTypeColor, pruneEmptyValues } from "@/lib/utils";
import { MarkerType } from "@xyflow/react";
import { default as NextLink } from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { useQueryClient } from "@tanstack/react-query";
import { getGetV2ListLibraryAgentsQueryKey } from "@/app/api/__generated__/endpoints/library/library";

export default function useAgentGraph(
  flowID?: GraphID,
  flowVersion?: number,
  flowExecutionID?: GraphExecutionID,
  passDataToBeads?: boolean,
) {
  const { toast } = useToast();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const api = useBackendAPI();
  const queryClient = useQueryClient();

  const [isScheduling, setIsScheduling] = useState(false);
  const [savedAgent, setSavedAgent] = useState<Graph | null>(null);
  const [agentDescription, setAgentDescription] = useState<string>("");
  const [agentName, setAgentName] = useState<string>("");
  const [libraryAgent, setLibraryAgent] = useState<LibraryAgent | null>(null);
  const [agentRecommendedScheduleCron, setAgentRecommendedScheduleCron] =
    useState<string>("");
  const [allBlocks, setAllBlocks] = useState<Block[]>([]);
  const [availableFlows, setAvailableFlows] = useState<GraphMeta[]>([]);
  const [updateQueue, setUpdateQueue] = useState<NodeExecutionResult[]>([]);
  const processedUpdates = useRef<NodeExecutionResult[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [activeExecutionID, setActiveExecutionID] =
    useState<GraphExecutionID | null>(null);
  const [graphExecutionError, setGraphExecutionError] = useState<string | null>(
    null,
  );
  const [xyNodes, setXYNodes] = useState<CustomNode[]>([]);
  const [xyEdges, setXYEdges] = useState<CustomEdge[]>([]);
  const { state, completeStep, incrementRuns } = useOnboarding();
  const betaBlocks = useGetFlag(Flag.BETA_BLOCKS);

  // Filter blocks based on beta flags
  const availableBlocks = useMemo(() => {
    return allBlocks.filter(
      (block) => !betaBlocks || !betaBlocks.includes(block.id),
    );
  }, [allBlocks, betaBlocks]);

  // Load available blocks & flows (stable - only loads once)
  useEffect(() => {
    api
      .getBlocks()
      .then((blocks) => {
        setAllBlocks(blocks);
      })
      .catch();

    api
      .listGraphs()
      .then((flows) => setAvailableFlows(flows))
      .catch();
  }, [api]);

  // Subscribe to execution events
  useEffect(() => {
    const deregisterMessageHandler = api.onWebSocketMessage(
      "node_execution_event",
      (data) => {
        if (data.graph_exec_id != flowExecutionID) {
          return;
        }
        setUpdateQueue((prev) => [...prev, data]);
      },
    );

    const deregisterConnectHandler =
      flowID && flowExecutionID
        ? api.onWebSocketConnect(() => {
            // Subscribe to execution updates
            api
              .subscribeToGraphExecution(flowExecutionID)
              .then(() =>
                console.debug(
                  `Subscribed to updates for execution #${flowExecutionID}`,
                ),
              )
              .catch((error) =>
                console.error(
                  `Failed to subscribe to updates for execution #${flowExecutionID}:`,
                  error,
                ),
              );

            // Sync execution info to ensure it's up-to-date after (re)connect
            api
              .getGraphExecutionInfo(flowID, flowExecutionID)
              .then((execution) =>
                setUpdateQueue((prev) => {
                  if (!execution.node_executions) return prev;
                  return [...prev, ...execution.node_executions];
                }),
              );
          })
        : () => {};

    return () => {
      deregisterMessageHandler();
      deregisterConnectHandler();
    };
  }, [api, flowID, flowExecutionID]);

  const getOutputType = useCallback(
    (nodes: CustomNode[], nodeId: string, handleId: string) => {
      const node = nodes.find((n) => n.id === nodeId);
      if (!node) return "unknown";

      const outputSchema = node.data.outputSchema;
      if (!outputSchema) return "unknown";

      const outputHandle = outputSchema.properties[handleId] || {};
      if (!("type" in outputHandle)) return "unknown";
      return outputHandle.type;
    },
    [],
  );

  // Load existing graph
  function _loadGraph(graph: Graph) {
    setSavedAgent(graph);
    setAgentName(graph.name);
    setAgentDescription(graph.description);
    setAgentRecommendedScheduleCron(graph.recommended_schedule_cron || "");

    setXYNodes((prevNodes) => {
      const _newNodes = graph.nodes.map((node) =>
        _backendNodeToXYNode(
          node,
          graph,
          prevNodes.find((n) => n.id === node.id),
        ),
      );
      const newNodes = _newNodes.filter((n) => n !== null);
      setXYEdges(() =>
        graph.links.map((link) => {
          const adjustedSourceName = cleanupSourceName(link.source_name);
          return {
            id: formatEdgeID(link),
            type: "custom",
            data: {
              edgeColor: getTypeColor(
                getOutputType(newNodes, link.source_id, adjustedSourceName!),
              ),
              sourcePos: newNodes.find((node) => node.id === link.source_id)
                ?.position,
              isStatic: link.is_static,
              beadUp: 0,
              beadDown: 0,
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              strokeWidth: 2,
              color: getTypeColor(
                getOutputType(newNodes, link.source_id, adjustedSourceName!),
              ),
            },
            source: link.source_id,
            target: link.sink_id,
            sourceHandle: adjustedSourceName || undefined,
            targetHandle: link.sink_name || undefined,
          };
        }),
      );
      return newNodes;
    });
  }

  function _backendNodeToXYNode(
    node: Node,
    graph: Graph,
    prevNode?: CustomNode,
  ): CustomNode | null {
    const block = availableBlocks.find((block) => block.id === node.block_id)!;
    if (!block) return null;

    const { position, ...metadata } = node.metadata;
    const subGraphName =
      (block.uiType == BlockUIType.AGENT && _getSubGraphName(node)) || null;

    return {
      id: node.id,
      type: "custom",
      position: {
        x: position?.x || 0,
        y: position?.y || 0,
      },
      data: {
        isOutputOpen: false,
        ...prevNode?.data,
        block_id: block.id,
        blockType: subGraphName || block.name,
        blockCosts: block.costs,
        categories: block.categories,
        description: block.description,
        title: `${block.name} ${node.id}`,
        inputSchema: block.inputSchema,
        outputSchema: block.outputSchema,
        hardcodedValues: node.input_default,
        uiType: block.uiType,
        metadata: metadata,
        connections: graph.links
          .filter((l) => [l.source_id, l.sink_id].includes(node.id))
          .map((link) => ({
            edge_id: formatEdgeID(link),
            source: link.source_id,
            sourceHandle: link.source_name,
            target: link.sink_id,
            targetHandle: link.sink_name,
          })),
        backend_id: node.id,
      },
    };
  }

  const _getSubGraphName = (node: Node) => {
    if (node.input_default.agent_name) {
      return node.input_default.agent_name;
    }
    return (
      availableFlows.find((flow) => flow.id === node.input_default.graph_id)
        ?.name || null
    );
  };

  const getFrontendId = useCallback(
    (backendId: string, nodes: CustomNode[]) => {
      const node = nodes.find((node) => node.data.backend_id === backendId);
      return node?.id;
    },
    [],
  );

  /** --- Smart Decision Maker Block helper functions --- */

  const isToolSourceName = (sourceName: string) =>
    sourceName.startsWith("tools_^_");

  const cleanupSourceName = (sourceName: string) =>
    isToolSourceName(sourceName) ? "tools" : sourceName;

  const getToolFuncName = useCallback(
    (nodeID: string) => {
      const sinkNode = xyNodes.find((node) => node.id === nodeID);
      if (!sinkNode) return "";

      const sinkNodeName =
        sinkNode.data.block_id === SpecialBlockID.AGENT
          ? sinkNode.data.hardcodedValues?.agent_name ||
            availableFlows.find(
              (flow) => flow.id === sinkNode.data.hardcodedValues.graph_id,
            )?.name ||
            "agentexecutorblock"
          : sinkNode.data.title.split(" ")[0];

      return sinkNodeName;
    },
    [xyNodes, availableFlows],
  );

  const normalizeToolName = (str: string) =>
    str.replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase(); // This normalization rule has to match with the one on smart_decision_maker.py

  /** ------------------------------ */

  const updateEdgeBeads = useCallback(
    (nodeExecUpdate: NodeExecutionResult) => {
      setXYEdges((edges) =>
        edges.map((edge): CustomEdge => {
          if (edge.target !== getFrontendId(nodeExecUpdate.node_id, xyNodes)) {
            // If the edge does not match the target node, skip it
            return edge;
          }

          const execStatuses =
            edge.data!.beadData ??
            new Map<string, NodeExecutionResult["status"]>();

          if (edge.targetHandle! in nodeExecUpdate.input_data) {
            // If the execution event has an input from this edge, store its status
            execStatuses.set(
              nodeExecUpdate.node_exec_id,
              nodeExecUpdate.status,
            );
          }

          // Calculate bead counts based on execution status
          // eslint-disable-next-line prefer-const
          let [beadUp, beadDown] = execStatuses.values().reduce(
            ([beadUp, beadDown], status) => [
              beadUp + 1,
              // Count any non-incomplete execution as consumed
              status !== "INCOMPLETE" ? beadDown + 1 : beadDown,
            ],
            [0, 0],
          );

          // For static edges, ensure beadUp is always beadDown + 1
          // This is because static edges represent reusable inputs that are never fully consumed
          // The +1 represents the input that's still available for reuse
          if (edge.data?.isStatic && beadUp > 0) {
            beadUp = beadDown + 1;
          }

          // Update edge data
          return {
            ...edge,
            data: {
              ...edge.data!,
              beadUp,
              beadDown,
              beadData: new Map(execStatuses.entries()),
            },
          };
        }),
      );
    },
    [getFrontendId, xyNodes],
  );

  const addExecutionDataToNode = useCallback(
    (node: CustomNode, executionData: NodeExecutionResult) => {
      if (!executionData.output_data) {
        console.warn(
          `Execution data for node ${executionData.node_id} is empty, skipping update`,
        );
        return node;
      }

      const executionResults = [
        // Execution updates are not cumulative, so we need to filter out the old ones.
        ...(node.data.executionResults?.filter(
          (result) => result.execId !== executionData.node_exec_id,
        ) || []),
        {
          execId: executionData.node_exec_id,
          data: {
            "[Input]": [executionData.input_data],
            ...executionData.output_data,
          },
          status: executionData.status,
        },
      ];

      const statusRank = {
        RUNNING: 0,
        QUEUED: 1,
        INCOMPLETE: 2,
        TERMINATED: 3,
        COMPLETED: 4,
        FAILED: 5,
      };
      const status = executionResults
        .map((v) => v.status)
        .reduce((a, b) => (statusRank[a] < statusRank[b] ? a : b));

      return {
        ...node,
        data: {
          ...node.data,
          status,
          executionResults,
          isOutputOpen: true,
        },
      };
    },
    [],
  );

  const updateNodesWithExecutionData = useCallback(
    (executionData: NodeExecutionResult) => {
      if (!executionData.node_id) return;
      if (passDataToBeads) {
        updateEdgeBeads(executionData);
      }
      setXYNodes((nodes) => {
        const nodeId = nodes.find(
          (node) => node.data.backend_id === executionData.node_id,
        )?.id;
        if (!nodeId) {
          console.error(
            "Node not found for execution data:",
            executionData,
            "This shouldn't happen and means that the frontend and backend are out of sync.",
          );
          return nodes;
        }
        return nodes.map((node) =>
          node.id === nodeId
            ? addExecutionDataToNode(node, executionData)
            : node,
        );
      });
    },
    [passDataToBeads, updateEdgeBeads],
  );

  // Load graph
  useEffect(() => {
    if (!flowID || availableBlocks.length == 0) return;
    if (savedAgent?.id === flowID && savedAgent.version === flowVersion) return;

    api.getGraph(flowID, flowVersion).then((graph) => {
      console.debug("Fetching graph", flowID, "version", flowVersion);
      if (graph.version === savedAgent?.version) return; // in case flowVersion is not set

      console.debug("Loading graph", graph.id, "version", graph.version);
      _loadGraph(graph);
    });
  }, [flowID, flowVersion, availableBlocks, api]);

  // Load library agent
  useEffect(() => {
    if (!flowID) return;
    api
      .getLibraryAgentByGraphID(flowID, flowVersion)
      .then((libraryAgent) => setLibraryAgent(libraryAgent))
      .catch((error) => {
        console.warn(
          `Failed to fetch LibraryAgent for graph #${flowID} v${flowVersion}`,
          error,
        );
      });
  }, [api, flowID, flowVersion]);

  // Check if local graph state is in sync with backend
  const nodesSyncedWithSavedAgent = useMemo(() => {
    if (!savedAgent || xyNodes.length === 0) return false;

    // Check that all nodes of the saved agent are included in the current state
    return savedAgent.nodes.every((savedNode) =>
      xyNodes.find((node) => node.data.backend_id === savedNode.id),
    );
  }, [savedAgent, xyNodes]);

  // Update nodes with execution data
  useEffect(() => {
    if (updateQueue.length === 0 || !nodesSyncedWithSavedAgent) {
      return;
    }
    setUpdateQueue((prev) => {
      prev.forEach((data) => {
        updateNodesWithExecutionData(data);
        // Execution updates are not cumulative, so we need to filter out the old ones.
        processedUpdates.current = processedUpdates.current.filter(
          (update) => update.node_exec_id !== data.node_exec_id,
        );
        processedUpdates.current.push(data);
      });
      return [];
    });
  }, [updateQueue, nodesSyncedWithSavedAgent, updateNodesWithExecutionData]);

  useEffect(() => {
    if (!flowID || !flowExecutionID) {
      return;
    }

    const fetchExecutions = async () => {
      const execution = await api.getGraphExecutionInfo(
        flowID,
        flowExecutionID,
      );

      // Set graph execution error from the initial fetch
      if (execution.status === "FAILED") {
        setGraphExecutionError(
          execution.stats?.error ||
            "The execution failed due to an internal error. You can re-run the agent to retry.",
        );
      }

      if (
        (execution.status === "QUEUED" || execution.status === "RUNNING") &&
        !isRunning
      ) {
        setIsRunning(true);
        setActiveExecutionID(flowExecutionID);
      }
      setUpdateQueue((prev) => {
        if (!execution.node_executions) return prev;
        return [...prev, ...execution.node_executions];
      });

      const cancelGraphExecListener = api.onWebSocketMessage(
        "graph_execution_event",
        (graphExec) => {
          if (graphExec.id != flowExecutionID) {
            return;
          }

          // Update graph execution error state and show toast
          if (graphExec.status === "FAILED") {
            const errorMessage =
              graphExec.stats?.error ||
              "The execution failed due to an internal error. You can re-run the agent to retry.";
            setGraphExecutionError(errorMessage);

            if (
              graphExec.stats?.error
                ?.toLowerCase()
                .includes("insufficient balance")
            ) {
              // Show no credits toast if user has low credits
              toast({
                variant: "destructive",
                title: "Credits low",
                description: (
                  <div>
                    Agent execution failed due to insufficient credits.
                    <br />
                    Go to the{" "}
                    <NextLink
                      className="text-purple-300"
                      href="/profile/credits"
                    >
                      Credits
                    </NextLink>{" "}
                    page to top up.
                  </div>
                ),
                duration: 5000,
              });
            } else {
              // Show general graph execution error
              toast({
                variant: "destructive",
                title: "Agent execution failed",
                description: errorMessage,
                duration: 8000,
              });
            }
          }
          if (
            graphExec.status === "COMPLETED" ||
            graphExec.status === "TERMINATED" ||
            graphExec.status === "FAILED"
          ) {
            cancelGraphExecListener();
            setIsRunning(false);
            setIsStopping(false);
            setActiveExecutionID(null);
            incrementRuns();
          }
        },
      );
    };

    fetchExecutions();
  }, [flowID, flowExecutionID, incrementRuns]);

  const prepareNodeInputData = useCallback(
    (node: CustomNode) => {
      const blockSchema = availableBlocks.find(
        (n) => n.id === node.data.block_id,
      )?.inputSchema;

      if (!blockSchema) {
        console.error(`Schema not found for block ID: ${node.data.block_id}`);
        return {};
      }

      return rebuildObjectUsingSchema(
        blockSchema,
        pruneEmptyValues(node.data.hardcodedValues),
      );
    },
    [availableBlocks],
  );

  const prepareSaveableGraph = useCallback((): GraphCreatable => {
    const links = xyEdges.map((edge): LinkCreatable => {
      let sourceName = edge.sourceHandle || "";
      const sourceNode = xyNodes.find((node) => node.id === edge.source);
      const sinkNode = xyNodes.find((node) => node.id === edge.target);

      // Special case for SmartDecisionMakerBlock
      if (
        sourceNode?.data.block_id === SpecialBlockID.SMART_DECISION &&
        sourceName.toLowerCase() === "tools"
      ) {
        sourceName = `tools_^_${normalizeToolName(getToolFuncName(edge.target))}_~_${normalizeToolName(edge.targetHandle || "")}`;
      }
      return {
        source_id: sourceNode?.data.backend_id ?? edge.source,
        sink_id: sinkNode?.data.backend_id ?? edge.target,
        source_name: sourceName,
        sink_name: edge.targetHandle || "",
      };
    });

    return {
      name: agentName || `New Agent ${new Date().toISOString()}`,
      description: agentDescription || "",
      recommended_schedule_cron: agentRecommendedScheduleCron || null,
      nodes: xyNodes.map(
        (node): NodeCreatable => ({
          id: node.data.backend_id ?? node.id,
          block_id: node.data.block_id,
          input_default: prepareNodeInputData(node),
          metadata: {
            ...(node.data.metadata ?? {}),
            position: node.position,
          },
        }),
      ),
      links: links,
    };
  }, [
    xyNodes,
    xyEdges,
    agentName,
    agentDescription,
    agentRecommendedScheduleCron,
    prepareNodeInputData,
    getToolFuncName,
  ]);

  const resetEdgeBeads = useCallback(() => {
    setXYEdges((edges) =>
      edges.map(
        (edge): CustomEdge => ({
          ...edge,
          data: {
            ...edge.data,
            edgeColor: edge.data?.edgeColor ?? "grey",
            beadUp: 0,
            beadDown: 0,
            beadData: new Map(),
          },
        }),
      ),
    );
  }, [setXYEdges]);

  const _saveAgent = useCallback(async () => {
    // FIXME: frontend IDs should be resolved better (e.g. returned from the server)
    // currently this relies on block_id and position
    const blockIDToNodeIDMap = Object.fromEntries(
      xyNodes.map((node) => [
        `${node.data.block_id}_${node.position.x}_${node.position.y}`,
        node.id,
      ]),
    );

    const payload = prepareSaveableGraph();

    // To avoid saving the same graph, we compare the payload with the saved agent.
    // Differences in IDs are ignored.
    let newSavedAgent: Graph;
    if (savedAgent && graphsEquivalent(savedAgent, payload)) {
      console.warn("No need to save: Graph is the same as version on server");
      resetEdgeBeads();
      return savedAgent;
    } else {
      console.debug(
        "Saving new Graph version; old vs new:",
        savedAgent,
        payload,
      );

      newSavedAgent = savedAgent
        ? await api.updateGraph(savedAgent.id, {
            ...payload,
            id: savedAgent.id,
          })
        : await api.createGraph(payload);

      console.debug("Response from the API:", newSavedAgent);
    }

    // Update the URL
    if (newSavedAgent.version !== savedAgent?.version) {
      const path = new URLSearchParams(searchParams);
      path.set("flowID", newSavedAgent.id);
      path.set("flowVersion", newSavedAgent.version.toString());
      router.push(`${pathname}?${path.toString()}`);
    }

    // Update the node IDs on the frontend
    setSavedAgent(newSavedAgent);

    setXYNodes((prev) =>
      newSavedAgent.nodes
        .map((backendNode) => {
          const key = `${backendNode.block_id}_${backendNode.metadata.position.x}_${backendNode.metadata.position.y}`;
          const frontendNodeID = blockIDToNodeIDMap[key];
          const frontendNode = prev.find((node) => node.id === frontendNodeID);

          const { position, ...metadata } = backendNode.metadata;
          return frontendNode
            ? ({
                ...frontendNode,
                position,
                data: {
                  ...frontendNode.data,
                  // NOTE: we don't update `node.id` because it would also require
                  //  updating many references in other places. Instead, we keep the
                  //  backend node ID in `node.data.backend_id`.
                  backend_id: backendNode.id,
                  metadata,

                  // Reset & close node output
                  isOutputOpen: false,
                  status: undefined,
                  executionResults: undefined,
                },
              } satisfies CustomNode)
            : _backendNodeToXYNode(backendNode, newSavedAgent); // fallback
        })
        .filter((node) => node !== null),
    );

    // Reset bead count
    resetEdgeBeads();
    return newSavedAgent;
  }, [
    api,
    xyNodes,
    agentName,
    agentDescription,
    prepareNodeInputData,
    prepareSaveableGraph,
    savedAgent,
    pathname,
    router,
    searchParams,
    resetEdgeBeads,
  ]);

  const saveAgent = useCallback(async () => {
    setIsSaving(true);
    try {
      await _saveAgent();

      await queryClient.invalidateQueries({
        queryKey: getGetV2ListLibraryAgentsQueryKey(),
      });

      completeStep("BUILDER_SAVE_AGENT");
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      console.error("Error saving agent", error);
      toast({
        variant: "destructive",
        title: "Error saving agent",
        description: errorMessage,
      });
    } finally {
      setIsSaving(false);
    }
  }, [_saveAgent, toast, completeStep]);

  const saveAndRun = useCallback(
    async (
      inputs: Record<string, any>,
      credentialsInputs: Record<string, CredentialsMetaInput>,
    ) => {
      if (isSaving || isRunning) {
        return;
      }

      setIsSaving(true);
      let savedAgent: Graph;
      try {
        savedAgent = await _saveAgent();
        completeStep("BUILDER_SAVE_AGENT");
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        toast({
          variant: "destructive",
          title: "Error saving agent",
          description: errorMessage,
        });
        return;
      } finally {
        setIsSaving(false);
      }

      // NOTE: Client-side validation is skipped here because the backend now provides
      // comprehensive validation that includes credentialsInputs, which the frontend
      // validation cannot access. The backend will return structured validation errors
      // if there are any issues.

      setIsRunning(true);
      processedUpdates.current = [];

      try {
        const graphExecution = await api.executeGraph(
          savedAgent.id,
          savedAgent.version,
          inputs,
          credentialsInputs,
        );

        setActiveExecutionID(graphExecution.id);

        // Update URL params
        const path = new URLSearchParams(searchParams);
        path.set("flowID", savedAgent.id);
        path.set("flowVersion", savedAgent.version.toString());
        path.set("flowExecutionID", graphExecution.id);
        router.push(`${pathname}?${path.toString()}`);

        if (state?.completedSteps.includes("BUILDER_SAVE_AGENT")) {
          completeStep("BUILDER_RUN_AGENT");
        }
      } catch (error) {
        // Check if this is a structured validation error from the backend
        if (error instanceof ApiError && error.isGraphValidationError()) {
          const errorData = error.response.detail;

          // 1. Apply validation errors to the corresponding nodes.
          // 2. Clear existing errors for nodes that don't have validation issues.
          setXYNodes((nodes) => {
            return nodes.map((node) => {
              const nodeErrors = node.data.backend_id
                ? (errorData.node_errors[node.data.backend_id] ?? {})
                : {};
              return {
                ...node,
                data: {
                  ...node.data,
                  errors: nodeErrors,
                },
              };
            });
          });

          // Show a general toast about validation errors
          toast({
            variant: "destructive",
            title: errorData.message || "Graph validation failed",
            description:
              "Please fix the validation errors on the highlighted nodes and try again.",
          });
          setIsRunning(false);
          setActiveExecutionID(null);
          return;
        }

        // Generic error handling for non-validation errors
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        toast({
          variant: "destructive",
          title: "Error running agent",
          description: errorMessage,
        });
        setIsRunning(false);
        setActiveExecutionID(null);
      }
    },
    [
      _saveAgent,
      toast,
      completeStep,
      api,
      searchParams,
      pathname,
      router,
      state,
      isSaving,
      isRunning,
      processedUpdates,
    ],
  );

  const stopRun = useCallback(async () => {
    if (!isRunning || !activeExecutionID || isStopping || !savedAgent) {
      return;
    }

    setIsStopping(true);
    await api
      .stopGraphExecution(savedAgent.id, activeExecutionID)
      .catch((error) => {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        toast({
          variant: "destructive",
          title: "Error stopping agent",
          description: errorMessage,
        });
      });
    setIsStopping(false);
  }, [api, savedAgent, activeExecutionID, isRunning, isStopping, toast]);

  // runs after saving cron expression and inputs (if exists)
  const createRunSchedule = useCallback(
    async (
      cronExpression: string,
      scheduleName: string,
      inputs: Record<string, any>,
      credentialsInputs: Record<string, CredentialsMetaInput>,
    ) => {
      if (!savedAgent || isScheduling) return;

      // Validate cron expression
      if (!cronExpression || cronExpression.trim() === "") {
        toast({
          variant: "destructive",
          title: "Invalid schedule",
          description: "Please enter a valid cron expression",
        });
        return;
      }

      setIsScheduling(true);
      try {
        await api.createGraphExecutionSchedule({
          graph_id: savedAgent.id,
          graph_version: savedAgent.version,
          name: scheduleName,
          cron: cronExpression,
          inputs: inputs,
          credentials: credentialsInputs,
        });
        toast({
          title: "Agent scheduling successful",
        });

        // if scheduling is done from the monitor page, then redirect to monitor page after successful scheduling
        if (searchParams.get("open_scheduling") === "true") {
          router.push("/monitoring");
        }
      } catch (error) {
        console.error("Error scheduling agent:", error);
        toast({
          variant: "destructive",
          title: "Error scheduling agent",
          description: "Please retry",
        });
      } finally {
        setIsScheduling(false);
      }
    },
    [api, savedAgent, isScheduling, toast, searchParams, router],
  );

  return {
    agentName,
    setAgentName,
    agentDescription,
    setAgentDescription,
    agentRecommendedScheduleCron,
    setAgentRecommendedScheduleCron,
    savedAgent,
    libraryAgent,
    availableBlocks,
    availableFlows,
    getOutputType,
    saveAgent,
    saveAndRun,
    stopRun,
    createRunSchedule,
    isSaving,
    isRunning,
    isStopping,
    isScheduling,
    graphExecutionError,
    nodes: xyNodes,
    setNodes: setXYNodes,
    edges: xyEdges,
    setEdges: setXYEdges,
  };
}

function graphsEquivalent(saved: Graph, current: GraphCreatable): boolean {
  const sortNodes = (nodes: NodeCreatable[]) =>
    nodes.toSorted((a, b) => a.id.localeCompare(b.id));

  const sortLinks = (links: LinkCreatable[]) =>
    links.toSorted(
      (a, b) =>
        8 * a.source_id.localeCompare(b.source_id) +
        4 * a.sink_id.localeCompare(b.sink_id) +
        2 * a.source_name.localeCompare(b.source_name) +
        a.sink_name.localeCompare(b.sink_name),
    );

  const _saved = {
    name: saved.name,
    description: saved.description,
    nodes: sortNodes(saved.nodes).map((v) => ({
      block_id: v.block_id,
      input_default: v.input_default,
      metadata: v.metadata,
    })),
    links: sortLinks(saved.links).map((v) => ({
      sink_name: v.sink_name,
      source_name: v.source_name,
    })),
  };
  const _current = {
    name: current.name,
    description: current.description,
    nodes: sortNodes(current.nodes).map(({ id: _, ...rest }) => rest),
    links: sortLinks(current.links).map(
      ({ source_id: _, sink_id: __, ...rest }) => rest,
    ),
  };
  return deepEquals(_saved, _current);
}

function rebuildObjectUsingSchema(
  schema: BlockIOSubSchema,
  object: { [key: string]: any },
): Record<string, any> {
  let inputData: Record<string, any> = {};

  if ("properties" in schema) {
    Object.keys(schema.properties).forEach((key) => {
      if (object[key] !== undefined) {
        if (
          "properties" in schema.properties[key] ||
          "additionalProperties" in schema.properties[key]
        ) {
          inputData[key] = rebuildObjectUsingSchema(
            schema.properties[key],
            object[key],
          );
        } else {
          inputData[key] = object[key];
        }
      }
    });
  }

  if ("additionalProperties" in schema) {
    inputData = { ...inputData, ...object };
  }

  return inputData;
}
