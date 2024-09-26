import { CustomEdge } from "@/components/CustomEdge";
import { CustomNode } from "@/components/CustomNode";
import AutoGPTServerAPI, {
  Block,
  BlockIOSubSchema,
  Graph,
  Link,
  NodeExecutionResult,
} from "@/lib/autogpt-server-api";
import {
  deepEquals,
  getTypeColor,
  removeEmptyStringsAndNulls,
  setNestedProperty,
} from "@/lib/utils";
import { Connection, MarkerType } from "@xyflow/react";
import Ajv from "ajv";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams, usePathname } from "next/navigation";

const ajv = new Ajv({ strict: false, allErrors: true });

export default function useAgentGraph(
  flowID?: string,
  template?: boolean,
  passDataToBeads?: boolean,
) {
  const [router, searchParams, pathname] = [
    useRouter(),
    useSearchParams(),
    usePathname(),
  ];
  const [savedAgent, setSavedAgent] = useState<Graph | null>(null);
  const [agentDescription, setAgentDescription] = useState<string>("");
  const [agentName, setAgentName] = useState<string>("");
  const [availableNodes, setAvailableNodes] = useState<Block[]>([]);
  const [updateQueue, setUpdateQueue] = useState<NodeExecutionResult[]>([]);
  const processedUpdates = useRef<NodeExecutionResult[]>([]);
  /**
   * User `request` to save or save&run the agent, or to stop the active run.
   * `state` is used to track the request status:
   * - none: no request
   * - saving: request was sent to save the agent
   *   and nodes are pending sync to update their backend ids
   * - running: request was sent to run the agent
   *   and frontend is enqueueing execution results
   * - stopping: a request to stop the active run has been sent; response is pending
   * - error: request failed
   */
  const [saveRunRequest, setSaveRunRequest] = useState<
    | {
        request: "none" | "save" | "run";
        state: "none" | "saving" | "error";
      }
    | {
        request: "run" | "stop";
        state: "running" | "stopping" | "error";
        activeExecutionID?: string;
      }
  >({
    request: "none",
    state: "none",
  });
  // Determines if nodes backend ids are synced with saved agent (actual ids on the backend)
  const [nodesSyncedWithSavedAgent, setNodesSyncedWithSavedAgent] =
    useState(false);
  const [nodes, setNodes] = useState<CustomNode[]>([]);
  const [edges, setEdges] = useState<CustomEdge[]>([]);

  const api = useMemo(
    () => new AutoGPTServerAPI(process.env.NEXT_PUBLIC_AGPT_SERVER_URL!),
    [],
  );

  // Connect to WebSocket
  useEffect(() => {
    api
      .connectWebSocket()
      .then(() => {
        console.debug("WebSocket connected");
        api.onWebSocketMessage("execution_event", (data) => {
          setUpdateQueue((prev) => [...prev, data]);
        });
      })
      .catch((error) => {
        console.error("Failed to connect WebSocket:", error);
      });

    return () => {
      api.disconnectWebSocket();
    };
  }, [api]);

  // Load available blocks
  useEffect(() => {
    api
      .getBlocks()
      .then((blocks) => setAvailableNodes(blocks))
      .catch();
  }, [api]);

  //TODO to utils? repeated in Flow
  const formatEdgeID = useCallback((conn: Link | Connection): string => {
    if ("sink_id" in conn) {
      return `${conn.source_id}_${conn.source_name}_${conn.sink_id}_${conn.sink_name}`;
    } else {
      return `${conn.source}_${conn.sourceHandle}_${conn.target}_${conn.targetHandle}`;
    }
  }, []);

  const getOutputType = useCallback(
    (nodes: CustomNode[], nodeId: string, handleId: string) => {
      const node = nodes.find((n) => n.id === nodeId);
      if (!node) return "unknown";

      const outputSchema = node.data.outputSchema;
      if (!outputSchema) return "unknown";

      const outputHandle = outputSchema.properties[handleId];
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

      setNodes(() => {
        const newNodes = graph.nodes.map((node) => {
          const block = availableNodes.find(
            (block) => block.id === node.block_id,
          )!;
          const newNode: CustomNode = {
            id: node.id,
            type: "custom",
            position: {
              x: node?.metadata?.position?.x || 0,
              y: node?.metadata?.position?.y || 0,
            },
            data: {
              block_id: block.id,
              blockType: block.name,
              blockCosts: block.costs,
              categories: block.categories,
              description: block.description,
              title: `${block.name} ${node.id}`,
              inputSchema: block.inputSchema,
              outputSchema: block.outputSchema,
              hardcodedValues: node.input_default,
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
              isOutputOpen: false,
            },
          };
          return newNode;
        });
        setEdges((_) =>
          graph.links.map((link) => ({
            id: formatEdgeID(link),
            type: "custom",
            data: {
              edgeColor: getTypeColor(
                getOutputType(newNodes, link.source_id, link.source_name!),
              ),
              sourcePos: newNodes.find((node) => node.id === link.source_id)
                ?.position,
              isStatic: link.is_static,
              beadUp: 0,
              beadDown: 0,
              beadData: [],
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              strokeWidth: 2,
              color: getTypeColor(
                getOutputType(newNodes, link.source_id, link.source_name!),
              ),
            },
            source: link.source_id,
            target: link.sink_id,
            sourceHandle: link.source_name || undefined,
            targetHandle: link.sink_name || undefined,
          })),
        );
        return newNodes;
      });
    },
    [availableNodes, formatEdgeID, getOutputType],
  );

  const getFrontendId = useCallback(
    (backendId: string, nodes: CustomNode[]) => {
      const node = nodes.find((node) => node.data.backend_id === backendId);
      return node?.id;
    },
    [],
  );

  const updateEdgeBeads = useCallback(
    (executionData: NodeExecutionResult) => {
      setEdges((edges) => {
        return edges.map((e) => {
          const edge = { ...e, data: { ...e.data } } as CustomEdge;

          if (executionData.status === "COMPLETED") {
            // Produce output beads
            for (let key in executionData.output_data) {
              if (
                edge.source !== getFrontendId(executionData.node_id, nodes) ||
                edge.sourceHandle !== key
              ) {
                continue;
              }
              edge.data!.beadUp = (edge.data!.beadUp ?? 0) + 1;
              // For static edges beadDown is always one less than beadUp
              // Because there's no queueing and one bead is always at the connection point
              if (edge.data?.isStatic) {
                edge.data!.beadDown = (edge.data!.beadUp ?? 0) - 1;
                edge.data!.beadData = edge.data!.beadData!.slice(0, -1);
                continue;
              }
              //todo kcze this assumes output at key is always array with one element
              edge.data!.beadData = [
                executionData.output_data[key][0],
                ...edge.data!.beadData!,
              ];
            }
          } else if (executionData.status === "RUNNING") {
            // Consume input beads
            for (let key in executionData.input_data) {
              if (
                edge.target !== getFrontendId(executionData.node_id, nodes) ||
                edge.targetHandle !== key
              ) {
                continue;
              }
              // Skip decreasing bead count if edge doesn't match or if it's static
              if (
                edge.data!.beadData![edge.data!.beadData!.length - 1] !==
                  executionData.input_data[key] ||
                edge.data?.isStatic
              ) {
                continue;
              }
              edge.data!.beadDown = (edge.data!.beadDown ?? 0) + 1;
              edge.data!.beadData = edge.data!.beadData!.slice(0, -1);
            }
          }
          return edge;
        });
      });
    },
    [getFrontendId, nodes],
  );

  const updateNodesWithExecutionData = useCallback(
    (executionData: NodeExecutionResult) => {
      if (passDataToBeads) {
        updateEdgeBeads(executionData);
      }
      setNodes((nodes) => {
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
            ? {
                ...node,
                data: {
                  ...node.data,
                  status: executionData.status,
                  executionResults:
                    Object.keys(executionData.output_data).length > 0
                      ? [
                          ...(node.data.executionResults || []),
                          {
                            execId: executionData.node_exec_id,
                            data: executionData.output_data,
                          },
                        ]
                      : node.data.executionResults,
                  isOutputOpen: true,
                },
              }
            : node,
        );
      });
    },
    [passDataToBeads, updateEdgeBeads],
  );

  useEffect(() => {
    if (!flowID || availableNodes.length == 0) return;

    (template ? api.getTemplate(flowID) : api.getGraph(flowID)).then(
      (graph) => {
        console.debug("Loading graph");
        loadGraph(graph);
      },
    );
  }, [flowID, template, availableNodes, api, loadGraph]);

  // Update nodes with execution data
  useEffect(() => {
    if (updateQueue.length === 0 || !nodesSyncedWithSavedAgent) {
      return;
    }
    setUpdateQueue((prev) => {
      prev.forEach((data) => {
        // Skip already processed updates by checking
        // if the data is in the processedUpdates array by reference
        // This is not to process twice in react dev mode
        // because it'll add double the beads
        if (processedUpdates.current.includes(data)) {
          return;
        }
        updateNodesWithExecutionData(data);
        processedUpdates.current.push(data);
      });
      return [];
    });
  }, [updateQueue, nodesSyncedWithSavedAgent, updateNodesWithExecutionData]);

  const validateNodes = useCallback((): boolean => {
    let isValid = true;

    nodes.forEach((node) => {
      const validate = ajv.compile(node.data.inputSchema);
      const errors = {} as { [key: string]: string };

      // Validate values against schema using AJV
      const valid = validate(node.data.hardcodedValues);
      if (!valid) {
        // Populate errors if validation fails
        validate.errors?.forEach((error) => {
          // Skip error if there's an edge connected
          const path =
            "dataPath" in error
              ? (error.dataPath as string)
              : error.instancePath;
          const handle = path.split(/[\/.]/)[0];
          if (
            node.data.connections.some(
              (conn) => conn.target === node.id || conn.targetHandle === handle,
            )
          ) {
            return;
          }
          console.warn("Error", error);
          isValid = false;
          if (path && error.message) {
            const key = path.slice(1);
            console.log("Error", key, error.message);
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
      // Set errors
      setNodes((nodes) => {
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

    return isValid;
  }, [nodes]);

  // Handle user requests
  useEffect(() => {
    // Ignore none request
    if (saveRunRequest.request === "none") {
      return;
    }
    // Display error message
    if (saveRunRequest.state === "error") {
      if (saveRunRequest.request === "save") {
        console.error("Error saving agent");
      } else if (saveRunRequest.request === "run") {
        console.error(`Error saving&running agent`);
      } else if (saveRunRequest.request === "stop") {
        console.error(`Error stopping agent`);
      }
      // Reset request
      setSaveRunRequest({
        request: "none",
        state: "none",
      });
      return;
    }
    // When saving request is done
    if (
      saveRunRequest.state === "saving" &&
      savedAgent &&
      nodesSyncedWithSavedAgent
    ) {
      // Reset request if only save was requested
      if (saveRunRequest.request === "save") {
        setSaveRunRequest({
          request: "none",
          state: "none",
        });
        // If run was requested, run the agent
      } else if (saveRunRequest.request === "run") {
        if (!validateNodes()) {
          console.error("Validation failed; aborting run");
          setSaveRunRequest({
            request: "none",
            state: "none",
          });
          return;
        }
        api.subscribeToExecution(savedAgent.id);
        setSaveRunRequest({ request: "run", state: "running" });
        api
          .executeGraph(savedAgent.id)
          .then((graphExecution) => {
            setSaveRunRequest({
              request: "run",
              state: "running",
              activeExecutionID: graphExecution.id,
            });

            // Track execution until completed
            const pendingNodeExecutions: Set<string> = new Set();
            const cancelExecListener = api.onWebSocketMessage(
              "execution_event",
              (nodeResult) => {
                // We are racing the server here, since we need the ID to filter events
                if (nodeResult.graph_exec_id != graphExecution.id) {
                  return;
                }
                if (
                  nodeResult.status != "COMPLETED" &&
                  nodeResult.status != "FAILED"
                ) {
                  pendingNodeExecutions.add(nodeResult.node_exec_id);
                } else {
                  pendingNodeExecutions.delete(nodeResult.node_exec_id);
                }
                if (pendingNodeExecutions.size == 0) {
                  // Assuming the first event is always a QUEUED node, and
                  // following nodes are QUEUED before all preceding nodes are COMPLETED,
                  // an empty set means the graph has finished running.
                  cancelExecListener();
                  setSaveRunRequest({ request: "none", state: "none" });
                }
              },
            );
          })
          .catch(() => setSaveRunRequest({ request: "run", state: "error" }));

        processedUpdates.current = processedUpdates.current = [];
      }
    }
    // Handle stop request
    if (
      saveRunRequest.request === "stop" &&
      saveRunRequest.state != "stopping" &&
      savedAgent &&
      saveRunRequest.activeExecutionID
    ) {
      setSaveRunRequest({
        request: "stop",
        state: "stopping",
        activeExecutionID: saveRunRequest.activeExecutionID,
      });
      api
        .stopGraphExecution(savedAgent.id, saveRunRequest.activeExecutionID)
        .then(() => setSaveRunRequest({ request: "none", state: "none" }));
    }
  }, [
    api,
    saveRunRequest,
    savedAgent,
    nodesSyncedWithSavedAgent,
    validateNodes,
  ]);

  // Check if node ids are synced with saved agent
  useEffect(() => {
    // Check if all node ids are synced with saved agent (frontend and backend)
    if (!savedAgent || nodes?.length === 0) {
      setNodesSyncedWithSavedAgent(false);
      return;
    }
    // Find at least one node that has backend id existing on any saved agent node
    // This will works as long as ALL ids are replaced each time the graph is run
    const oneNodeSynced = savedAgent.nodes.some(
      (backendNode) => backendNode.id === nodes[0].data.backend_id,
    );
    setNodesSyncedWithSavedAgent(oneNodeSynced);
  }, [savedAgent, nodes]);

  const prepareNodeInputData = useCallback(
    (node: CustomNode) => {
      console.debug(
        "Preparing input data for node:",
        node.id,
        node.data.blockType,
      );

      const blockSchema = availableNodes.find(
        (n) => n.id === node.data.block_id,
      )?.inputSchema;

      if (!blockSchema) {
        console.error(`Schema not found for block ID: ${node.data.block_id}`);
        return {};
      }

      const getNestedData = (
        schema: BlockIOSubSchema,
        values: { [key: string]: any },
      ): { [key: string]: any } => {
        let inputData: { [key: string]: any } = {};

        if ("properties" in schema) {
          Object.keys(schema.properties).forEach((key) => {
            if (values[key] !== undefined) {
              if (
                "properties" in schema.properties[key] ||
                "additionalProperties" in schema.properties[key]
              ) {
                inputData[key] = getNestedData(
                  schema.properties[key],
                  values[key],
                );
              } else {
                inputData[key] = values[key];
              }
            }
          });
        }

        if ("additionalProperties" in schema) {
          inputData = { ...inputData, ...values };
        }

        return inputData;
      };

      let inputData = getNestedData(blockSchema, node.data.hardcodedValues);

      console.debug(
        `Final prepared input for ${node.data.blockType} (${node.id}):`,
        inputData,
      );
      return inputData;
    },
    [availableNodes],
  );

  const saveAgent = useCallback(
    async (asTemplate: boolean = false) => {
      //FIXME frontend ids should be resolved better (e.g. returned from the server)
      // currently this relays on block_id and position
      const blockIdToNodeIdMap: Record<string, string> = {};

      nodes.forEach((node) => {
        const key = `${node.data.block_id}_${node.position.x}_${node.position.y}`;
        blockIdToNodeIdMap[key] = node.id;
      });

      const formattedNodes = nodes.map((node) => {
        const inputDefault = prepareNodeInputData(node);
        const inputNodes = edges
          .filter((edge) => edge.target === node.id)
          .map((edge) => ({
            name: edge.targetHandle || "",
            node_id: edge.source,
          }));

        const outputNodes = edges
          .filter((edge) => edge.source === node.id)
          .map((edge) => ({
            name: edge.sourceHandle || "",
            node_id: edge.target,
          }));

        return {
          id: node.id,
          block_id: node.data.block_id,
          input_default: inputDefault,
          input_nodes: inputNodes,
          output_nodes: outputNodes,
          data: {
            ...node.data,
            hardcodedValues: removeEmptyStringsAndNulls(
              node.data.hardcodedValues,
            ),
          },
          metadata: { position: node.position },
        };
      });

      const links = edges.map((edge) => ({
        source_id: edge.source,
        sink_id: edge.target,
        source_name: edge.sourceHandle || "",
        sink_name: edge.targetHandle || "",
      }));

      const payload = {
        id: savedAgent?.id!,
        name: agentName || "Agent Name",
        description: agentDescription || "Agent Description",
        nodes: formattedNodes,
        links: links,
      };

      // To avoid saving the same graph, we compare the payload with the saved agent.
      // Differences in IDs are ignored.
      const comparedPayload = {
        ...(({ id, ...rest }) => rest)(payload),
        nodes: payload.nodes.map(
          ({ id, data, input_nodes, output_nodes, ...rest }) => rest,
        ),
        links: payload.links.map(({ source_id, sink_id, ...rest }) => rest),
      };
      const comparedSavedAgent = {
        name: savedAgent?.name,
        description: savedAgent?.description,
        nodes: savedAgent?.nodes.map((v) => ({
          block_id: v.block_id,
          input_default: v.input_default,
          metadata: v.metadata,
        })),
        links: savedAgent?.links.map((v) => ({
          sink_name: v.sink_name,
          source_name: v.source_name,
        })),
      };

      let newSavedAgent = null;
      if (savedAgent && deepEquals(comparedPayload, comparedSavedAgent)) {
        console.warn("No need to save: Graph is the same as version on server");
        newSavedAgent = savedAgent;
      } else {
        console.debug(
          "Saving new Graph version; old vs new:",
          comparedPayload,
          payload,
        );
        setNodesSyncedWithSavedAgent(false);

        newSavedAgent = savedAgent
          ? await (savedAgent.is_template
              ? api.updateTemplate(savedAgent.id, payload)
              : api.updateGraph(savedAgent.id, payload))
          : await (asTemplate
              ? api.createTemplate(payload)
              : api.createGraph(payload));

        console.debug("Response from the API:", newSavedAgent);
      }

      // Route the URL to the new flow ID if it's a new agent.
      if (!savedAgent) {
        const path = new URLSearchParams(searchParams);
        path.set("flowID", newSavedAgent.id);
        router.push(`${pathname}?${path.toString()}`);
        return;
      }

      // Update the node IDs on the frontend
      setSavedAgent(newSavedAgent);
      setNodes((prev) => {
        return newSavedAgent.nodes
          .map((backendNode) => {
            const key = `${backendNode.block_id}_${backendNode.metadata.position.x}_${backendNode.metadata.position.y}`;
            const frontendNodeId = blockIdToNodeIdMap[key];
            const frontendNode = prev.find(
              (node) => node.id === frontendNodeId,
            );

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
                    executionResults: [],
                  },
                }
              : null;
          })
          .filter((node) => node !== null);
      });
      // Reset bead count
      setEdges((edges) => {
        return edges.map((edge) => ({
          ...edge,
          data: {
            ...edge.data,
            edgeColor: edge.data?.edgeColor!,
            beadUp: 0,
            beadDown: 0,
            beadData: [],
          },
        }));
      });
    },
    [
      api,
      nodes,
      edges,
      savedAgent,
      agentName,
      agentDescription,
      prepareNodeInputData,
    ],
  );

  const requestSave = useCallback(
    (asTemplate: boolean) => {
      saveAgent(asTemplate);
      setSaveRunRequest({
        request: "save",
        state: "saving",
      });
    },
    [saveAgent],
  );

  const requestSaveAndRun = useCallback(() => {
    saveAgent();
    setSaveRunRequest({
      request: "run",
      state: "saving",
    });
  }, [saveAgent]);

  const requestStopRun = useCallback(() => {
    if (saveRunRequest.state != "running") {
      return;
    }
    if (!saveRunRequest.activeExecutionID) {
      console.warn(
        "Stop requested but execution ID is unknown; state:",
        saveRunRequest,
      );
    }
    setSaveRunRequest((prev) => ({
      ...prev,
      request: "stop",
      state: "running",
    }));
  }, [saveRunRequest]);

  return {
    agentName,
    setAgentName,
    agentDescription,
    setAgentDescription,
    savedAgent,
    availableNodes,
    getOutputType,
    requestSave,
    requestSaveAndRun,
    requestStopRun,
    isSaving: saveRunRequest.state == "saving",
    isRunning: saveRunRequest.state == "running",
    isStopping: saveRunRequest.state == "stopping",
    nodes,
    setNodes,
    edges,
    setEdges,
  };
}
