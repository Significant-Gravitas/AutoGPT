"use client";
import React, {
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef,
} from "react";
import ReactFlow, {
  addEdge,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  OnConnect,
  NodeTypes,
  Connection,
  EdgeTypes,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import CustomNode, { CustomNodeData } from "./CustomNode";
import "./flow.css";
import AutoGPTServerAPI, {
  Block,
  BlockIOSchema,
  Graph,
  NodeExecutionResult,
} from "@/lib/autogpt-server-api";
import { Play, Undo2, Redo2 } from "lucide-react";
import {
  deepEquals,
  getTypeColor,
  removeEmptyStringsAndNulls,
  setNestedProperty,
} from "@/lib/utils";
import { history } from "./history";
import { CustomEdge, CustomEdgeData } from "./CustomEdge";
import ConnectionLine from "./ConnectionLine";
import Ajv from "ajv";
import { Control, ControlPanel } from "@/components/edit/control/ControlPanel";
import { SaveControl } from "@/components/edit/control/SaveControl";
import { BlocksControl } from "@/components/edit/control/BlocksControl";

// This is for the history, this is the minimum distance a block must move before it is logged
// It helps to prevent spamming the history with small movements especially when pressing on a input in a block
const MINIMUM_MOVE_BEFORE_LOG = 50;

const ajv = new Ajv({ strict: false, allErrors: true });

const FlowEditor: React.FC<{
  flowID?: string;
  template?: boolean;
  className?: string;
}> = ({ flowID, template, className }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState<CustomNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<CustomEdgeData>([]);
  const [nodeId, setNodeId] = useState<number>(1);
  const [availableNodes, setAvailableNodes] = useState<Block[]>([]);
  const [savedAgent, setSavedAgent] = useState<Graph | null>(null);
  const [agentDescription, setAgentDescription] = useState<string>("");
  const [agentName, setAgentName] = useState<string>("");
  const [copiedNodes, setCopiedNodes] = useState<Node<CustomNodeData>[]>([]);
  const [copiedEdges, setCopiedEdges] = useState<Edge<CustomEdgeData>[]>([]);
  const [isAnyModalOpen, setIsAnyModalOpen] = useState(false); // Track if any modal is open

  const apiUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL!;
  const api = useMemo(() => new AutoGPTServerAPI(apiUrl), [apiUrl]);
  const initialPositionRef = useRef<{
    [key: string]: { x: number; y: number };
  }>({});
  const isDragging = useRef(false);

  useEffect(() => {
    api
      .connectWebSocket()
      .then(() => {
        console.log("WebSocket connected");
        api.onWebSocketMessage("execution_event", (data) => {
          updateNodesWithExecutionData([data]);
        });
      })
      .catch((error) => {
        console.error("Failed to connect WebSocket:", error);
      });

    return () => {
      api.disconnectWebSocket();
    };
  }, [api]);

  useEffect(() => {
    api
      .getBlocks()
      .then((blocks) => setAvailableNodes(blocks))
      .catch();
  }, []);

  // Load existing graph
  useEffect(() => {
    if (!flowID || availableNodes.length == 0) return;

    (template ? api.getTemplate(flowID) : api.getGraph(flowID)).then((graph) =>
      loadGraph(graph),
    );
  }, [flowID, template, availableNodes]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
      const isUndo =
        (isMac ? event.metaKey : event.ctrlKey) && event.key === "z";
      const isRedo =
        (isMac ? event.metaKey : event.ctrlKey) &&
        (event.key === "y" || (event.shiftKey && event.key === "Z"));

      if (isUndo) {
        event.preventDefault();
        handleUndo();
      }

      if (isRedo) {
        event.preventDefault();
        handleRedo();
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const nodeTypes: NodeTypes = useMemo(() => ({ custom: CustomNode }), []);
  const edgeTypes: EdgeTypes = useMemo(() => ({ custom: CustomEdge }), []);

  const onNodesChangeStart = (event: MouseEvent, node: Node) => {
    initialPositionRef.current[node.id] = { ...node.position };
    isDragging.current = true;
  };

  const onNodesChangeEnd = (event: MouseEvent, node: Node | null) => {
    if (!node) return;

    isDragging.current = false;
    const oldPosition = initialPositionRef.current[node.id];
    const newPosition = node.position;

    // Calculate the movement distance
    if (!oldPosition || !newPosition) return;

    const distanceMoved = Math.sqrt(
      Math.pow(newPosition.x - oldPosition.x, 2) +
        Math.pow(newPosition.y - oldPosition.y, 2),
    );

    if (distanceMoved > MINIMUM_MOVE_BEFORE_LOG) {
      // Minimum movement threshold
      history.push({
        type: "UPDATE_NODE_POSITION",
        payload: { nodeId: node.id, oldPosition, newPosition },
        undo: () =>
          setNodes((nds) =>
            nds.map((n) =>
              n.id === node.id ? { ...n, position: oldPosition } : n,
            ),
          ),
        redo: () =>
          setNodes((nds) =>
            nds.map((n) =>
              n.id === node.id ? { ...n, position: newPosition } : n,
            ),
          ),
      });
    }
    delete initialPositionRef.current[node.id];
  };

  const updateNodesOnEdgeChange = (
    edge: Edge<CustomEdgeData>,
    action: "add" | "remove",
  ) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === edge.source || node.id === edge.target) {
          const connections =
            action === "add"
              ? [
                  ...node.data.connections,
                  {
                    source: edge.source,
                    sourceHandle: edge.sourceHandle!,
                    target: edge.target,
                    targetHandle: edge.targetHandle!,
                  },
                ]
              : node.data.connections.filter(
                  (conn) =>
                    !(
                      conn.source === edge.source &&
                      conn.target === edge.target &&
                      conn.sourceHandle === edge.sourceHandle &&
                      conn.targetHandle === edge.targetHandle
                    ),
                );
          return {
            ...node,
            data: {
              ...node.data,
              connections,
            },
          };
        }
        return node;
      }),
    );
  };

  const getOutputType = (id: string, handleId: string) => {
    const node = nodes.find((node) => node.id === id);
    if (!node) return "unknown";

    const outputSchema = node.data.outputSchema;
    if (!outputSchema) return "unknown";

    const outputHandle = outputSchema.properties[handleId];
    if (!("type" in outputHandle)) return "unknown";
    return outputHandle.type;
  };

  const getNodePos = (id: string) => {
    const node = nodes.find((node) => node.id === id);
    if (!node) return 0;

    return node.position;
  };

  // Function to clear status, output, and close the output info dropdown of all nodes
  const clearNodesStatusAndOutput = useCallback(() => {
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
          status: undefined,
          output_data: undefined,
          isOutputOpen: false, // Close the output info dropdown
        },
      })),
    );
  }, [setNodes]);

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      const edgeColor = getTypeColor(
        getOutputType(connection.source!, connection.sourceHandle!),
      );
      const sourcePos = getNodePos(connection.source!);
      console.log("sourcePos", sourcePos);
      const newEdge = {
        id: `${connection.source}_${connection.sourceHandle}_${connection.target}_${connection.targetHandle}`,
        type: "custom",
        markerEnd: {
          type: MarkerType.ArrowClosed,
          strokeWidth: 2,
          color: edgeColor,
        },
        data: { edgeColor, sourcePos },
        ...connection,
      };

      setEdges((eds) => {
        const newEdges = addEdge(newEdge, eds);
        history.push({
          type: "ADD_EDGE",
          payload: newEdge,
          undo: () => {
            setEdges((prevEdges) =>
              prevEdges.filter((edge) => edge.id !== newEdge.id),
            );
            updateNodesOnEdgeChange(newEdge, "remove");
          },
          redo: () => {
            setEdges((prevEdges) => addEdge(newEdge, prevEdges));
            updateNodesOnEdgeChange(newEdge, "add");
          },
        });
        updateNodesOnEdgeChange(newEdge, "add");
        return newEdges;
      });
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === connection.target || node.id === connection.source) {
            return {
              ...node,
              data: {
                ...node.data,
                connections: [
                  ...node.data.connections,
                  {
                    source: connection.source,
                    sourceHandle: connection.sourceHandle,
                    target: connection.target,
                    targetHandle: connection.targetHandle,
                  } as {
                    source: string;
                    sourceHandle: string;
                    target: string;
                    targetHandle: string;
                  },
                ],
              },
            };
          }
          return node;
        }),
      );
      clearNodesStatusAndOutput(); // Clear status and output on connection change
    },
    [nodes],
  );

  const onEdgesDelete = useCallback(
    (edgesToDelete: Edge<CustomEdgeData>[]) => {
      setNodes((nds) =>
        nds.map((node) => ({
          ...node,
          data: {
            ...node.data,
            connections: node.data.connections.filter(
              (conn: any) =>
                !edgesToDelete.some(
                  (edge) =>
                    edge.source === conn.source &&
                    edge.target === conn.target &&
                    edge.sourceHandle === edge.sourceHandle &&
                    edge.targetHandle === edge.targetHandle,
                ),
            ),
          },
        })),
      );
      clearNodesStatusAndOutput(); // Clear status and output on edge deletion
    },
    [setNodes, clearNodesStatusAndOutput],
  );

  const addNode = useCallback(
    (blockId: string, nodeType: string) => {
      const nodeSchema = availableNodes.find((node) => node.id === blockId);
      if (!nodeSchema) {
        console.error(`Schema not found for block ID: ${blockId}`);
        return;
      }

      const newNode: Node<CustomNodeData> = {
        id: nodeId.toString(),
        type: "custom",
        position: { x: Math.random() * 400, y: Math.random() * 400 },
        data: {
          blockType: nodeType,
          title: `${nodeType} ${nodeId}`,
          inputSchema: nodeSchema.inputSchema,
          outputSchema: nodeSchema.outputSchema,
          hardcodedValues: {},
          setHardcodedValues: (values) => {
            setNodes((nds) =>
              nds.map((node) =>
                node.id === newNode.id
                  ? { ...node, data: { ...node.data, hardcodedValues: values } }
                  : node,
              ),
            );
          },
          connections: [],
          isOutputOpen: false,
          block_id: blockId,
          setIsAnyModalOpen: setIsAnyModalOpen, // Pass setIsAnyModalOpen function
          setErrors: (errors: { [key: string]: string | null }) => {
            setNodes((nds) =>
              nds.map((node) =>
                node.id === newNode.id
                  ? { ...node, data: { ...node.data, errors } }
                  : node,
              ),
            );
          },
        },
      };

      setNodes((nds) => [...nds, newNode]);
      setNodeId((prevId) => prevId + 1);
      clearNodesStatusAndOutput(); // Clear status and output when a new node is added

      history.push({
        type: "ADD_NODE",
        payload: newNode,
        undo: () =>
          setNodes((nds) => nds.filter((node) => node.id !== newNode.id)),
        redo: () => setNodes((nds) => [...nds, newNode]),
      });
    },
    [nodeId, availableNodes],
  );

  const handleUndo = () => {
    history.undo();
  };

  const handleRedo = () => {
    history.redo();
  };

  function loadGraph(graph: Graph) {
    setSavedAgent(graph);
    setAgentName(graph.name);
    setAgentDescription(graph.description);

    setNodes(
      graph.nodes.map((node) => {
        const block = availableNodes.find(
          (block) => block.id === node.block_id,
        )!;
        const newNode: Node<CustomNodeData> = {
          id: node.id,
          type: "custom",
          position: {
            x: node.metadata.position.x,
            y: node.metadata.position.y,
          },
          data: {
            setIsAnyModalOpen: setIsAnyModalOpen,
            block_id: block.id,
            blockType: block.name,
            title: `${block.name} ${node.id}`,
            inputSchema: block.inputSchema,
            outputSchema: block.outputSchema,
            hardcodedValues: node.input_default,
            setHardcodedValues: (values: { [key: string]: any }) => {
              setNodes((nds) =>
                nds.map((node) =>
                  node.id === newNode.id
                    ? {
                        ...node,
                        data: { ...node.data, hardcodedValues: values },
                      }
                    : node,
                ),
              );
            },
            connections: graph.links
              .filter((l) => [l.source_id, l.sink_id].includes(node.id))
              .map((link) => ({
                source: link.source_id,
                sourceHandle: link.source_name,
                target: link.sink_id,
                targetHandle: link.sink_name,
              })),
            isOutputOpen: false,
            setIsAnyModalOpen: setIsAnyModalOpen, // Pass setIsAnyModalOpen function
            setErrors: (errors: { [key: string]: string | null }) => {
              setNodes((nds) =>
                nds.map((node) =>
                  node.id === newNode.id
                    ? { ...node, data: { ...node.data, errors } }
                    : node,
                ),
              );
            },
          },
        };
        return newNode;
      }),
    );

    setEdges(
      graph.links.map(
        (link) =>
          ({
            id: `${link.source_id}_${link.source_name}_${link.sink_id}_${link.sink_name}`,
            type: "custom",
            data: {
              edgeColor: getTypeColor(
                getOutputType(link.source_id, link.source_name!),
              ),
              sourcePos: getNodePos(link.source_id),
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              strokeWidth: 2,
              color: getTypeColor(
                getOutputType(link.source_id, link.source_name!),
              ),
            },
            source: link.source_id,
            target: link.sink_id,
            sourceHandle: link.source_name || undefined,
            targetHandle: link.sink_name || undefined,
          }) as Edge<CustomEdgeData>,
      ),
    );
  }

  const prepareNodeInputData = (
    node: Node<CustomNodeData>,
    allNodes: Node<CustomNodeData>[],
    allEdges: Edge<CustomEdgeData>[],
  ) => {
    console.log("Preparing input data for node:", node.id, node.data.blockType);

    const blockSchema = availableNodes.find(
      (n) => n.id === node.data.block_id,
    )?.inputSchema;

    if (!blockSchema) {
      console.error(`Schema not found for block ID: ${node.data.block_id}`);
      return {};
    }

    const getNestedData = (
      schema: BlockIOSchema,
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

    console.log(
      `Final prepared input for ${node.data.blockType} (${node.id}):`,
      inputData,
    );
    return inputData;
  };

  async function saveAgent(asTemplate: boolean = false) {
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
          hardcodedValues: removeEmptyStringsAndNulls(
            node.data.hardcodedValues,
          ),
          status: undefined,
        },
      })),
    );
    await new Promise((resolve) => setTimeout(resolve, 100));
    console.log("All nodes before formatting:", nodes);
    const blockIdToNodeIdMap = {};

    const formattedNodes = nodes.map((node) => {
      nodes.forEach((node) => {
        const key = `${node.data.block_id}_${node.position.x}_${node.position.y}`;
        blockIdToNodeIdMap[key] = node.id;
      });
      const inputDefault = prepareNodeInputData(node, nodes, edges);
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
      links: links, // Ensure this field is included
    };

    if (savedAgent && deepEquals(payload, savedAgent)) {
      console.debug("No need to save: Graph is the same as version on server");
      return;
    } else {
      console.debug(
        "Saving new Graph version; old vs new:",
        savedAgent,
        payload,
      );
    }

    const newSavedAgent = savedAgent
      ? await (savedAgent.is_template
          ? api.updateTemplate(savedAgent.id, payload)
          : api.updateGraph(savedAgent.id, payload))
      : await (asTemplate
          ? api.createTemplate(payload)
          : api.createGraph(payload));
    console.debug("Response from the API:", newSavedAgent);
    setSavedAgent(newSavedAgent);

    // Update the node IDs in the frontend
    const updatedNodes = newSavedAgent.nodes
      .map((backendNode) => {
        const key = `${backendNode.block_id}_${backendNode.metadata.position.x}_${backendNode.metadata.position.y}`;
        const frontendNodeId = blockIdToNodeIdMap[key];
        const frontendNode = nodes.find((node) => node.id === frontendNodeId);

        return frontendNode
          ? {
              ...frontendNode,
              position: backendNode.metadata.position,
              data: {
                ...frontendNode.data,
                backend_id: backendNode.id,
              },
            }
          : null;
      })
      .filter((node) => node !== null);

    setNodes(updatedNodes);

    return newSavedAgent.id;
  }

  const validateNodes = (): boolean => {
    let isValid = true;

    nodes.forEach((node) => {
      const validate = ajv.compile(node.data.inputSchema);
      const errors = {} as { [key: string]: string | null };

      // Validate values against schema using AJV
      const valid = validate(node.data.hardcodedValues);
      if (!valid) {
        // Populate errors if validation fails
        validate.errors?.forEach((error) => {
          // Skip error if there's an edge connected
          const path = error.instancePath || error.schemaPath;
          const handle = path.split(/[\/.]/)[0];
          if (
            node.data.connections.some(
              (conn) => conn.target === node.id || conn.targetHandle === handle,
            )
          ) {
            return;
          }
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
      node.data.setErrors(errors);
    });

    return isValid;
  };

  const runAgent = async () => {
    try {
      const newAgentId = await saveAgent();
      if (!newAgentId) {
        console.error("Error saving agent; aborting run");
        return;
      }

      if (!validateNodes()) {
        console.error("Validation failed; aborting run");
        return;
      }

      api.subscribeToExecution(newAgentId);
      api.runGraph(newAgentId);
    } catch (error) {
      console.error("Error running agent:", error);
    }
  };

  const updateNodesWithExecutionData = (
    executionData: NodeExecutionResult[],
  ) => {
    setNodes((nds) =>
      nds.map((node) => {
        const nodeExecution = executionData.find(
          (exec) => exec.node_id === node.data.backend_id,
        );
        if (nodeExecution) {
          return {
            ...node,
            data: {
              ...node.data,
              status: nodeExecution.status,
              output_data: nodeExecution.output_data,
              isOutputOpen: true,
            },
          };
        }
        return node;
      }),
    );
  };

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (isAnyModalOpen) return; // Prevent copy/paste if any modal is open

      if (event.ctrlKey || event.metaKey) {
        if (event.key === "c" || event.key === "C") {
          // Copy selected nodes
          const selectedNodes = nodes.filter((node) => node.selected);
          const selectedEdges = edges.filter((edge) => edge.selected);
          setCopiedNodes(selectedNodes);
          setCopiedEdges(selectedEdges);
        }
        if (event.key === "v" || event.key === "V") {
          // Paste copied nodes
          if (copiedNodes.length > 0) {
            const newNodes = copiedNodes.map((node, index) => {
              const newNodeId = (nodeId + index).toString();
              return {
                ...node,
                id: newNodeId,
                position: {
                  x: node.position.x + 20, // Offset pasted nodes
                  y: node.position.y + 20,
                },
                data: {
                  ...node.data,
                  status: undefined, // Reset status
                  output_data: undefined, // Clear output data
                  setHardcodedValues: (values: { [key: string]: any }) => {
                    setNodes((nds) =>
                      nds.map((n) =>
                        n.id === newNodeId
                          ? {
                              ...n,
                              data: { ...n.data, hardcodedValues: values },
                            }
                          : n,
                      ),
                    );
                  },
                },
              };
            });
            const updatedNodes = nodes.map((node) => ({
              ...node,
              selected: false,
            })); // Deselect old nodes
            setNodes([...updatedNodes, ...newNodes]);
            setNodeId((prevId) => prevId + copiedNodes.length);

            const newEdges = copiedEdges.map((edge) => {
              const newSourceId =
                newNodes.find((n) => n.data.title === edge.source)?.id ||
                edge.source;
              const newTargetId =
                newNodes.find((n) => n.data.title === edge.target)?.id ||
                edge.target;
              return {
                ...edge,
                id: `${newSourceId}_${edge.sourceHandle}_${newTargetId}_${edge.targetHandle}_${Date.now()}`,
                source: newSourceId,
                target: newTargetId,
              };
            });
            setEdges([...edges, ...newEdges]);
          }
        }
      }
    },
    [nodes, edges, copiedNodes, copiedEdges, nodeId, isAnyModalOpen],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleKeyDown]);

  const onNodesDelete = useCallback(() => {
    clearNodesStatusAndOutput();
  }, [clearNodesStatusAndOutput]);

  const editorControls: Control[] = [
    {
      label: "Undo",
      icon: <Undo2 />,
      onClick: handleUndo,
    },
    {
      label: "Redo",
      icon: <Redo2 />,
      onClick: handleRedo,
    },
    {
      label: "Run",
      icon: <Play />,
      onClick: runAgent,
    },
  ];

  return (
    <div className={className}>
      <ReactFlow
        nodes={nodes.map((node) => ({
          ...node,
          data: { ...node.data, setIsAnyModalOpen },
        }))}
        edges={edges.map((edge) => ({
          ...edge,
          data: { ...edge.data, clearNodesStatusAndOutput },
        }))}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        connectionLineComponent={ConnectionLine}
        onNodesDelete={onNodesDelete}
        onEdgesDelete={onEdgesDelete}
        deleteKeyCode={["Backspace", "Delete"]}
        onNodeDragStart={onNodesChangeStart}
        onNodeDragStop={onNodesChangeEnd}
      >
        <div className={"flex flex-row absolute z-10 gap-2"}>
          <ControlPanel controls={editorControls}>
            <BlocksControl blocks={availableNodes} addBlock={addNode} />
            <SaveControl
              agentMeta={savedAgent}
              onSave={saveAgent}
              onDescriptionChange={setAgentDescription}
              onNameChange={setAgentName}
            />
          </ControlPanel>
        </div>
      </ReactFlow>
    </div>
  );
};

export default FlowEditor;
