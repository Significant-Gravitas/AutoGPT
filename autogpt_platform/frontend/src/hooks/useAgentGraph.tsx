import { CustomEdge } from "@/components/CustomEdge";
import { CustomNode } from "@/components/CustomNode";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import { useToast } from "@/components/molecules/Toast/use-toast";
import BackendAPI, {
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
  LinkCreatable,
  NodeCreatable,
  NodeExecutionResult,
  SpecialBlockID,
} from "@/lib/autogpt-server-api";
import {
  deepEquals,
  getTypeColor,
  removeEmptyStringsAndNulls,
  setNestedProperty,
} from "@/lib/utils";
import { MarkerType } from "@xyflow/react";
import Ajv from "ajv";
import { default as NextLink } from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const ajv = new Ajv({ strict: false, allErrors: true });

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

  const [isScheduling, setIsScheduling] = useState(false);
  const [savedAgent, setSavedAgent] = useState<Graph | null>(null);
  const [agentDescription, setAgentDescription] = useState<string>("");
  const [agentName, setAgentName] = useState<string>("");
  const [availableBlocks, setAvailableBlocks] = useState<Block[]>([]);
  const [availableFlows, setAvailableFlows] = useState<GraphMeta[]>([]);
  const [updateQueue, setUpdateQueue] = useState<NodeExecutionResult[]>([]);
  const processedUpdates = useRef<NodeExecutionResult[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [activeExecutionID, setActiveExecutionID] =
    useState<GraphExecutionID | null>(null);
  const [xyNodes, setXYNodes] = useState<CustomNode[]>([]);
  const [xyEdges, setXYEdges] = useState<CustomEdge[]>([]);
  const { state, completeStep, incrementRuns } = useOnboarding();

  const api = useMemo(
    () => new BackendAPI(process.env.NEXT_PUBLIC_AGPT_SERVER_URL!),
    [],
  );

  // Load available blocks & flows
  useEffect(() => {
    api
      .getBlocks()
      .then((blocks) => setAvailableBlocks(blocks))
      .catch();

    api
      .listGraphs()
      .then((flows) => setAvailableFlows(flows))
      .catch();

    api.connectWebSocket().catch((error) => {
      console.error("Failed to connect WebSocket:", error);
    });

    return () => {
      api.disconnectWebSocket();
    };
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
  const loadGraph = useCallback(
    (graph: Graph) => {
      setSavedAgent(graph);
      setAgentName(graph.name);
      setAgentDescription(graph.description);

      setXYNodes((prevNodes) => {
        const _newNodes = graph.nodes.map((node) => {
          const block = availableBlocks.find(
            (block) => block.id === node.block_id,
          )!;
          if (!block) return null;
          const prevNode = prevNodes.find((n) => n.id === node.id);
          const flow =
            block.uiType == BlockUIType.AGENT
              ? availableFlows.find(
                  (flow) => flow.id === node.input_default.graph_id,
                )
              : null;
          const newNode: CustomNode = {
            id: node.id,
            type: "custom",
            position: {
              x: node?.metadata?.position?.x || 0,
              y: node?.metadata?.position?.y || 0,
            },
            data: {
              isOutputOpen: false,
              ...prevNode?.data,
              block_id: block.id,
              blockType: flow?.name || block.name,
              blockCosts: block.costs,
              categories: block.categories,
              description: block.description,
              title: `${block.name} ${node.id}`,
              inputSchema: block.inputSchema,
              outputSchema: block.outputSchema,
              hardcodedValues: node.input_default,
              webhook: node.webhook,
              uiType: block.uiType,
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
          return newNode;
        });
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
    },
    [availableBlocks, availableFlows, getOutputType],
  );

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
      const sinkNodeName = sinkNode
        ? sinkNode.data.block_id === SpecialBlockID.AGENT
          ? sinkNode.data.hardcodedValues?.graph_id
            ? availableFlows.find(
                (flow) => flow.id === sinkNode.data.hardcodedValues.graph_id,
              )?.name || "agentexecutorblock"
            : "agentexecutorblock"
          : sinkNode.data.title.split(" ")[0]
        : "";

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
      console.debug("Loading graph");
      loadGraph(graph);
    });
  }, [flowID, flowVersion, availableBlocks, api, loadGraph]);

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
          if (
            graphExec.status === "FAILED" &&
            graphExec?.stats?.error
              ?.toLowerCase()
              ?.includes("insufficient balance")
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
                    href="/marketplace/credits"
                  >
                    Credits
                  </NextLink>{" "}
                  page to top up.
                </div>
              ),
              duration: 5000,
            });
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

      return rebuildObjectUsingSchema(blockSchema, node.data.hardcodedValues);
    },
    [availableBlocks],
  );

  const prepareSaveableGraph = useCallback((): GraphCreatable => {
    const links = xyEdges.map((edge): LinkCreatable => {
      let sourceName = edge.sourceHandle || "";
      const sourceNode = xyNodes.find((node) => node.id === edge.source);

      // Special case for SmartDecisionMakerBlock
      if (
        sourceNode?.data.block_id === SpecialBlockID.SMART_DECISION &&
        sourceName.toLowerCase() === "tools"
      ) {
        sourceName = `tools_^_${normalizeToolName(getToolFuncName(edge.target))}_~_${normalizeToolName(edge.targetHandle || "")}`;
      }
      return {
        source_id: edge.source,
        sink_id: edge.target,
        source_name: sourceName,
        sink_name: edge.targetHandle || "",
      };
    });

    return {
      name: agentName || `New Agent ${new Date().toISOString()}`,
      description: agentDescription || "",
      nodes: xyNodes.map(
        (node): NodeCreatable => ({
          id: node.id,
          block_id: node.data.block_id,
          input_default: prepareNodeInputData(node),
          metadata: { position: node.position },
        }),
      ),
      links: links,
    };
  }, [
    xyNodes,
    xyEdges,
    agentName,
    agentDescription,
    prepareNodeInputData,
    getToolFuncName,
  ]);

  const validateGraph = useCallback(
    (graph?: GraphCreatable): string | null => {
      let errorMessage = null;

      if (!graph) {
        graph = prepareSaveableGraph();
      }

      graph.nodes.forEach((node) => {
        const block = availableBlocks.find(
          (block) => block.id === node.block_id,
        );
        if (!block) {
          console.error(
            `Node ${node.id} is invalid: unknown block ID ${node.block_id}`,
          );
          return;
        }
        const inputSchema = block.inputSchema;
        const validate = ajv.compile(inputSchema);
        const errors: Record<string, string> = {};
        const errorPrefix = `${block.name} [${node.id.split("-")[0]}]`;

        // Validate values against schema using AJV
        const inputData = node.input_default;
        const valid = validate(inputData);
        if (!valid) {
          // Populate errors if validation fails
          validate.errors?.forEach((error) => {
            const path =
              "dataPath" in error
                ? (error.dataPath as string)
                : error.instancePath || error.params.missingProperty;
            const handle = path.split(/[\/.]/)[0];
            // Skip error if there's an edge connected
            if (
              graph.links.some(
                (link) => link.sink_id == node.id && link.sink_name == handle,
              )
            ) {
              return;
            }
            console.warn(`Error in ${block.name} input: ${error}`, {
              data: inputData,
              schema: inputSchema,
            });
            errorMessage =
              `${errorPrefix}: ` + (error.message || "Invalid input");
            if (path && error.message) {
              const key = path.slice(1);
              setNestedProperty(
                errors,
                key,
                error.message[0].toUpperCase() + error.message.slice(1),
              );
            } else if (error.keyword === "required") {
              const key = error.params.missingProperty;
              setNestedProperty(errors, key, "This field is required");
            }
          });
        }

        Object.entries(inputSchema.properties).forEach(([key, schema]) => {
          if (schema.depends_on) {
            const dependencies = schema.depends_on;

            // Check if dependent field has value
            const hasValue =
              inputData[key] != null ||
              ("default" in schema && schema.default != null);

            const mustHaveValue = inputSchema.required?.includes(key);

            // Check for missing dependencies when dependent field is present
            const missingDependencies = dependencies.filter(
              (dep) =>
                !inputData[dep as keyof typeof inputData] ||
                String(inputData[dep as keyof typeof inputData]).trim() === "",
            );

            if ((hasValue || mustHaveValue) && missingDependencies.length > 0) {
              setNestedProperty(
                errors,
                key,
                `Requires ${missingDependencies.join(", ")} to be set`,
              );
              errorMessage = `${errorPrefix}: field ${key} requires ${missingDependencies.join(", ")} to be set`;
            }

            // Check if field is required when dependencies are present
            const hasAllDependencies = dependencies.every(
              (dep) =>
                inputData[dep as keyof typeof inputData] &&
                String(inputData[dep as keyof typeof inputData]).trim() !== "",
            );

            if (hasAllDependencies && !hasValue) {
              setNestedProperty(
                errors,
                key,
                `${key} is required when ${dependencies.join(", ")} are set`,
              );
              errorMessage = `${errorPrefix}: ${key} is required when ${dependencies.join(", ")} are set`;
            }
          }
        });

        // Set errors
        setXYNodes((nodes) => {
          return nodes.map((n) => {
            if (n.id === node.id) {
              return {
                ...n,
                data: {
                  ...n.data,
                  errors,
                },
              };
            }
            return n;
          });
        });
      });

      return errorMessage;
    },
    [prepareSaveableGraph, availableBlocks],
  );

  const _saveAgent = useCallback(async () => {
    // FIXME: frontend IDs should be resolved better (e.g. returned from the server)
    // currently this relays on block_id and position
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
      newSavedAgent = savedAgent;
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
    setXYNodes((prev) => {
      return newSavedAgent.nodes
        .map((backendNode) => {
          const key = `${backendNode.block_id}_${backendNode.metadata.position.x}_${backendNode.metadata.position.y}`;
          const frontendNodeID = blockIDToNodeIDMap[key];
          const frontendNode = prev.find((node) => node.id === frontendNodeID);

          return frontendNode
            ? {
                ...frontendNode,
                position: backendNode.metadata.position,
                data: {
                  ...frontendNode.data,
                  hardcodedValues: removeEmptyStringsAndNulls(
                    frontendNode.data.hardcodedValues,
                  ),
                  status: undefined,
                  backend_id: backendNode.id,
                  webhook: backendNode.webhook,
                  executionResults: [],
                },
              }
            : null;
        })
        .filter((node) => node !== null);
    });
    // Reset bead count
    setXYEdges((edges) => {
      return edges.map(
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
      );
    });
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
  ]);

  const saveAgent = useCallback(async () => {
    setIsSaving(true);
    try {
      await _saveAgent();
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

      const validationError = validateGraph(savedAgent);
      if (validationError) {
        toast({
          title: `Graph validation failed: ${validationError}`,
          variant: "destructive",
        });
        return;
      }

      setIsRunning(true);
      processedUpdates.current = [];

      try {
        const graphExecution = await api.executeGraph(
          savedAgent.id,
          savedAgent.version,
          inputs,
          credentialsInputs,
        );

        setActiveExecutionID(graphExecution.graph_exec_id);

        // Update URL params
        const path = new URLSearchParams(searchParams);
        path.set("flowID", savedAgent.id);
        path.set("flowVersion", savedAgent.version.toString());
        path.set("flowExecutionID", graphExecution.graph_exec_id);
        router.push(`${pathname}?${path.toString()}`);

        if (state?.completedSteps.includes("BUILDER_SAVE_AGENT")) {
          completeStep("BUILDER_RUN_AGENT");
        }
      } catch (error) {
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
      savedAgent,
      prepareSaveableGraph,
      nodesSyncedWithSavedAgent,
      validateGraph,
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
        console.error(error);
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
    savedAgent,
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
    nodes: xyNodes,
    setNodes: setXYNodes,
    edges: xyEdges,
    setEdges: setXYEdges,
  };
}

function graphsEquivalent(saved: Graph, current: GraphCreatable): boolean {
  const _saved = {
    name: saved.name,
    description: saved.description,
    nodes: saved.nodes.map((v) => ({
      block_id: v.block_id,
      input_default: v.input_default,
      metadata: v.metadata,
    })),
    links: saved.links.map((v) => ({
      sink_name: v.sink_name,
      source_name: v.source_name,
    })),
  };
  const _current = {
    name: current.name,
    description: current.description,
    nodes: current.nodes.map(({ id: _, ...rest }) => rest),
    links: current.links.map(({ source_id: _, sink_id: __, ...rest }) => rest),
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
