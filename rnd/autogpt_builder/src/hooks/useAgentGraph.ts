import { CustomEdge } from "@/components/CustomEdge";
import { CustomNode } from "@/components/CustomNode";
import AutoGPTServerAPI, { Block, BlockIOSubSchema, Graph, Link, NodeExecutionResult } from "@/lib/autogpt-server-api";
import { deepEquals, getTypeColor, removeEmptyStringsAndNulls } from "@/lib/utils";
import { Connection, MarkerType } from "@xyflow/react";
import { use, useEffect, useMemo, useRef, useState } from "react";

export default function useAgentGraph(flowID?: string, template?: boolean) {
  const [savedAgent, setSavedAgent] = useState<Graph | null>(null);
  const [agentDescription, setAgentDescription] = useState<string>("");
  const [agentName, setAgentName] = useState<string>("");
  const [availableNodes, setAvailableNodes] = useState<Block[]>([]);
  const [saveRunRequest, setSaveRunRequest] = useState<{
    request: 'none' | 'save' | 'run'
    state: 'none' | 'saving' | 'running' | 'error'
    prevAgentVersion: number | null
  }>({
    request: 'none',
    state: 'none',
    prevAgentVersion: null
  });
  const [nodesSyncedWithSavedAgent, setNodesSyncedWithSavedAgent] = useState(false);
  // const [timestamps, setTimestamps] = useState<{

  // }>({});
  const [nodes, setNodes] = useState<CustomNode[]>([]);
  const [edges, setEdges] = useState<CustomEdge[]>([]);

  // const [state, setState] = useState<{
  //   agent: Graph | null,
  //   nodes: CustomNode[],
  //   edges: CustomEdge[],
  // }>({ agent: null, nodes: [], edges: [] });

  // const { agent: savedAgent, nodes, edges } = state;

  const apiUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL!;
  const api = useMemo(() => new AutoGPTServerAPI(apiUrl), [apiUrl]);

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
    // Ignore none request
    if (saveRunRequest.request === 'none') {
      return;
    }
    // Display error message
    if (saveRunRequest.state === 'error') {
      if (saveRunRequest.request === 'save') {
        console.error("Error saving agent");
      } else if (saveRunRequest.request === 'run') {
        console.error(`Error saving&running agent`);
      }
      // Reset request
      setSaveRunRequest((prev) => ({
        ...prev,
        request: 'none',
        state: 'none',
        // prevAgentVersion: null
      }));
      return;
    }
    // When saving request is done
    if (saveRunRequest.state === 'saving' && savedAgent && nodesSyncedWithSavedAgent) {
      console.log("### Agent saved & synced with frontend id:", savedAgent.id);

      // Reset request if only save was requested
      if (saveRunRequest.request === 'save') {
        setSaveRunRequest((prev) => ({
          ...prev,
          request: 'none',
          state: 'none',
          // prevAgentVersion: null
        }));
        // If run was requested, run the agent
      } else if (saveRunRequest.request === 'run') {
        //todo kcze validate nodes
        // if (!validateNodes()) {
        //   console.error("Validation failed; aborting run");
        //   setSaveRunRequest({
        //     request: 'none',
        //     state: 'none'
        //   });
        //   return;
        // }
        setNodesSyncedWithSavedAgent(false);
        api.subscribeToExecution(savedAgent.id);
        api.executeGraph(savedAgent.id);

        setSaveRunRequest((prev) => ({
          ...prev,
          request: 'run',
          state: 'running',
          // prevAgentVersion: savedAgent.version
        }));
      }
    }

  }, [saveRunRequest, savedAgent, nodes, nodesSyncedWithSavedAgent]);

  // function isNodesSyncedWithSavedAgent() {
  //   if (!savedAgent || nodes?.length === 0) {
  //     return false;
  //   }
  //   if (savedAgent.id === saveRunRequest.prevAgentId) {
  //     return false;
  //   }
  //   // return nodes.every(node => savedAgent.nodes.some(backendNode => backendNode.id === node.data.backend_id));
  //   //todo kcze this should be enough but check
  //   return savedAgent.nodes.some(backendNode => backendNode.id === nodes[0].data.backend_id);
  // }


  useEffect(() => {
    // Check if all node ids are synced with saved agent (frontend and backend)
    if (!savedAgent || nodes?.length === 0) {
      setNodesSyncedWithSavedAgent(false);
      console.log("NO agent or NO nodes");
      return;
    }
    // if (savedAgent.version === saveRunRequest.prevAgentVersion) {
    //   setNodesSyncedWithSavedAgent(false);
    //   console.log("Agent version is the same as previous request");
    //   return;
    // }
    // Find at least one node that has backend_id existing on any saved agent node
    // const allNodesSynced = nodes.every(node => savedAgent.nodes.some(backendNode => backendNode.id === node.data.backend_id));
    //todo kcze this should be enough but check
    const oneNodeSynced = savedAgent.nodes.some(backendNode => backendNode.id === nodes[0].data.backend_id);
    setNodesSyncedWithSavedAgent(oneNodeSynced);
    console.log("±±±±± Nodes synced with saved agent:", oneNodeSynced);
    // console.log(`node 0 id: ${nodes[0].data.backend_id}`)
    // console.log(`node 1 id: ${nodes[1].data.backend_id}`)
  }, [savedAgent, nodes]);

  // function setNodes(setter: (nodes: CustomNode[]) => CustomNode[]) {
  //   setState((prev) => ({
  //     agent: prev.agent,
  //     nodes: setter(prev.nodes),
  //     edges: prev.edges,
  //   }));
  // }

  // function setEdges(setter: (edges: CustomEdge[]) => CustomEdge[]) {
  //   setState((prev) => ({
  //     agent: prev.agent,
  //     nodes: prev.nodes,
  //     edges: setter(prev.edges),
  //   }));
  // }

  function getFrontendId(backendId: string, nodes: CustomNode[]) {
    const node = nodes.find((node) => node.data.backend_id === backendId);
    return node?.id;
  }

  function updateEdgeBeads(
    executionData: NodeExecutionResult[],
    nodes: CustomNode[],
  ) {
    setEdges((edges) => {
      // console.log("££ Updating edges with execution data", executionData);
      const newEdges = JSON.parse(
        JSON.stringify(edges),
      ) as CustomEdge[];

      executionData.forEach((exec) => {
        if (exec.status === "COMPLETED") {
          // Produce output beads
          for (let key in exec.output_data) {
            const outputEdges = newEdges.filter(
              (edge) =>
                edge.source === getFrontendId(exec.node_id, nodes) &&
                edge.sourceHandle === key,
            );
            outputEdges.forEach((edge) => {
              edge.data!.beadUp = (edge.data!.beadUp ?? 0) + 1;
              // For static edges beadDown is always one less than beadUp
              // Because there's no queueing and one bead is always at the connection point
              if (edge.data?.isStatic) {
                edge.data!.beadDown = (edge.data!.beadUp ?? 0) - 1;
                edge.data!.beadData! = edge.data!.beadData!.slice(0, -1);
              }
              //todo kcze this assumes output at key is always array with one element
              edge.data!.beadData = [
                exec.output_data[key][0],
                ...edge.data!.beadData!,
              ];
            });
          }
        } else if (exec.status === "RUNNING") {
          // Consume input beads
          for (let key in exec.input_data) {
            const inputEdges = newEdges.filter(
              (edge) =>
                edge.target === getFrontendId(exec.node_id, nodes) &&
                edge.targetHandle === key,
            );

            inputEdges.forEach((edge) => {
              // Skip decreasing bead count if edge doesn't match or if it's static
              if (
                edge.data!.beadData![edge.data!.beadData!.length - 1] !==
                exec.input_data[key] ||
                edge.data?.isStatic
              ) {
                return;
              }
              console.log('edge id', edge.id)
              console.log("Consuming beadDown:", edge.data!.beadDown);
              edge.data!.beadDown = (edge.data!.beadDown ?? 0) + 1;
              edge.data!.beadData! = edge.data!.beadData!.slice(0, -1);
            });
          }
        }
      });

      return newEdges;
    });
  }

  const updateNodesWithExecutionData = (
    executionData: NodeExecutionResult[],
  ) => {
    console.log("Updating nodes with execution data:", executionData);
    console.table({
      // graph_id: executionData[0].graph_id,
      node_id: executionData[0].node_id,
    })
    //todo kcze turning off beads
    // if (visualizeBeads !== "no") {
    // updateEdgeBeads(executionData, nodes);
    // }
    setNodes((nodes) => {
      // console.log('NoDeS on execution update', nodes)
      const updatedNodes = nodes.map((node) => {

        const nodeExecution = executionData.find(
          (exec) => {
            return exec.node_id === node.data.backend_id
          }
        );
        // console.log('exec.node_id', executionData[0].node_id, 'nodeExecution', nodeExecution)

        if (!nodeExecution || node.data.status === nodeExecution.status) {
          return node;
        }

        return {
          ...node,
          data: {
            ...node.data,
            status: nodeExecution.status,
            output_data: nodeExecution.output_data,
            isOutputOpen: true,
          },
        };
      });

      return updatedNodes;
    });
  };

  //kcze to utils? repeated in Flow
  function formatEdgeID(conn: Link | Connection): string {
    if ("sink_id" in conn) {
      return `${conn.source_id}_${conn.source_name}_${conn.sink_id}_${conn.sink_name}`;
    } else {
      return `${conn.source}_${conn.sourceHandle}_${conn.target}_${conn.targetHandle}`;
    }
  }

  const getOutputType = (nodeId: string, handleId: string) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return "unknown";

    const outputSchema = node.data.outputSchema;
    if (!outputSchema) return "unknown";

    const outputHandle = outputSchema.properties[handleId];
    if (!("type" in outputHandle)) return "unknown";
    return outputHandle.type;
  };

  function loadGraph(graph: Graph) {
    setSavedAgent(graph);
    //todo kcze pack all three into setState
    // setState((prev) => ({
    //   agent: graph,
    //   nodes: prev.nodes,
    //   edges: prev.edges,
    // }));
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
            x: node.metadata.position.x,
            y: node.metadata.position.y,
          },
          data: {
            block_id: block.id,
            blockType: block.name,
            categories: block.categories,
            description: block.description,
            title: `${block.name} ${node.id}`,
            inputSchema: block.inputSchema,
            outputSchema: block.outputSchema,
            hardcodedValues: node.input_default,
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
        graph.links.map(
          (link) =>
          ({
            id: formatEdgeID(link),
            type: "custom",
            data: {
              edgeColor: getTypeColor(
                getOutputType(link.source_id, link.source_name!),
              ),
              sourcePos: nodes.find(node => node.id === link.source_id)?.position,
              isStatic: link.is_static,
              beadUp: 0,
              beadDown: 0,
              beadData: [],
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
          }),
        ),
      );
      return newNodes;
    });
  }

  const prepareNodeInputData = (node: CustomNode) => {
    console.debug("Preparing input data for node:", node.id, node.data.blockType);

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
  };

  const saveAgent = async (asTemplate: boolean = false) => {
    // setNodes((nds) =>
    //   nds.map((node) => ({
    //     ...node,
    //     data: {
    //       ...node.data,
    //       hardcodedValues: removeEmptyStringsAndNulls(
    //         node.data.hardcodedValues,
    //       ),
    //       status: undefined,
    //       // backend_id: undefined
    //     },
    //   })),
    // );
    
    //todo kcze frontend ids should be returned from the server
    // currently this relays on block_id and position
    const blockIdToNodeIdMap: Record<string, string> = {};

    const formattedNodes = nodes.map((node) => {
      //todo kcze move outside of map!!
      nodes.forEach((node) => {
        const key = `${node.data.block_id}_${node.position.x}_${node.position.y}`;
        blockIdToNodeIdMap[key] = node.id;
      });
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

    // if (savedAgent && deepEquals(payload, savedAgent)) {
    //   console.debug("No need to save: Graph is the same as version on server");
    //   // Trigger state change
    //   setSavedAgent(savedAgent);
    //   // setState((prev) => prev);
    //   return;
    // } else {
    //   console.debug(
    //     "Saving new Graph version; old vs new:",
    //     savedAgent,
    //     payload,
    //   );
    // }

    setNodesSyncedWithSavedAgent(false);

    const newSavedAgent = savedAgent
      ? await (savedAgent.is_template
        ? api.updateTemplate(savedAgent.id, payload)
        : api.updateGraph(savedAgent.id, payload))
      : await (asTemplate
        ? api.createTemplate(payload)
        : api.createGraph(payload));
    console.debug("Response from the API:", newSavedAgent);
    console.table({
      // graph_id: newSavedAgent.id,
      node_0_id: newSavedAgent.nodes[0].id,
      node_1_id: newSavedAgent.nodes[1].id,
    })

    
    // setState((prev) => ({
    //   agent: newSavedAgent,
    //   nodes: newSavedAgent.nodes
    //     .map((backendNode) => {
    //       const key = `${backendNode.block_id}_${backendNode.metadata.position.x}_${backendNode.metadata.position.y}`;
    //       const frontendNodeId = blockIdToNodeIdMap[key];
    //       const frontendNode = prev.nodes.find((node) => node.id === frontendNodeId);

    //       return frontendNode
    //         ? {
    //           ...frontendNode,
    //           position: backendNode.metadata.position,
    //           data: {
    //             ...frontendNode.data,
    //             backend_id: backendNode.id,
    //           },
    //         }
    //         : null;
    //     })
    //     .filter((node) => node !== null),
    //   edges: prev.edges,
    // }));
    // Update the node IDs on the frontend
    setSavedAgent(newSavedAgent);
    setNodes((prev) => {

      return newSavedAgent.nodes
        .map((backendNode) => {
          const key = `${backendNode.block_id}_${backendNode.metadata.position.x}_${backendNode.metadata.position.y}`;
          const frontendNodeId = blockIdToNodeIdMap[key];
          const frontendNode = prev.find((node) => node.id === frontendNodeId);

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
              },
            }
            : null;
        })
        .filter((node) => node !== null);
    });
    // Reset bead count
    setEdges((edges) => {
      return edges.map(
        (edge) =>
        ({
          ...edge,
          data: {
            ...edge.data,
            edgeColor: edge.data?.edgeColor!,
            beadUp: 0,
            beadDown: 0,
            beadData: [],
          },
        }),
      );
    });
  }

  const requestSave = (asTemplate: boolean) => {
    saveAgent(asTemplate);
    setSaveRunRequest((prev) => {
      // if (prev.request !== 'none') {
      //   return prev;
      // }

      return {
        request: 'save',
        state: 'saving',
        prevAgentVersion: savedAgent?.version ?? null,
      }
    })
  }

  const requestSaveRun = () => {
    saveAgent();
    setSaveRunRequest((prev) => {
      // if (prev.request !== 'none') {
      //   return prev;
      // }

      return {
        request: 'run',
        state: 'saving',
        prevAgentVersion: savedAgent?.version ?? null,
      }
    })
  }

  return {
    agentName,
    setAgentName,
    agentDescription,
    setAgentDescription,
    savedAgent,
    availableNodes,
    getOutputType,
    requestSave,
    requestSaveRun,
    nodes,
    setNodes,
    edges,
    setEdges,
  }

}