"use client";
import React, {
  useState,
  useCallback,
  useEffect,
  useRef,
  MouseEvent,
  createContext,
} from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  Background,
  Node,
  OnConnect,
  Connection,
  MarkerType,
  NodeChange,
  EdgeChange,
  useReactFlow,
  applyEdgeChanges,
  applyNodeChanges,
  useViewport,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { CustomNode } from "./CustomNode";
import "./flow.css";
import { Link } from "@/lib/autogpt-server-api";
import { getTypeColor, filterBlocksByType } from "@/lib/utils";
import { history } from "./history";
import { CustomEdge } from "./CustomEdge";
import ConnectionLine from "./ConnectionLine";
import { Control, ControlPanel } from "@/components/edit/control/ControlPanel";
import { SaveControl } from "@/components/edit/control/SaveControl";
import { BlocksControl } from "@/components/edit/control/BlocksControl";
import { IconUndo2, IconRedo2 } from "@/components/ui/icons";
import { startTutorial } from "./tutorial";
import useAgentGraph from "@/hooks/useAgentGraph";
import { v4 as uuidv4 } from "uuid";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import RunnerUIWrapper, {
  RunnerUIWrapperRef,
} from "@/components/RunnerUIWrapper";
import PrimaryActionBar from "@/components/PrimaryActionButton";
import { useToast } from "@/components/ui/use-toast";

// This is for the history, this is the minimum distance a block must move before it is logged
// It helps to prevent spamming the history with small movements especially when pressing on a input in a block
const MINIMUM_MOVE_BEFORE_LOG = 50;

type FlowContextType = {
  visualizeBeads: "no" | "static" | "animate";
  setIsAnyModalOpen: (isOpen: boolean) => void;
  getNextNodeId: () => string;
};

export const FlowContext = createContext<FlowContextType | null>(null);

const FlowEditor: React.FC<{
  flowID?: string;
  template?: boolean;
  className?: string;
}> = ({ flowID, template, className }) => {
  const { addNodes, addEdges, getNode, deleteElements, updateNode } =
    useReactFlow<CustomNode, CustomEdge>();
  const [nodeId, setNodeId] = useState<number>(1);
  const [copiedNodes, setCopiedNodes] = useState<CustomNode[]>([]);
  const [copiedEdges, setCopiedEdges] = useState<CustomEdge[]>([]);
  const [isAnyModalOpen, setIsAnyModalOpen] = useState(false);
  const [visualizeBeads, setVisualizeBeads] = useState<
    "no" | "static" | "animate"
  >("animate");
  const {
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
    isRunning,
    nodes,
    setNodes,
    edges,
    setEdges,
  } = useAgentGraph(flowID, template, visualizeBeads !== "no");

  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const initialPositionRef = useRef<{
    [key: string]: { x: number; y: number };
  }>({});
  const isDragging = useRef(false);

  // State to control if blocks menu should be pinned open
  const [pinBlocksPopover, setPinBlocksPopover] = useState(false);
  // State to control if save popover should be pinned open
  const [pinSavePopover, setPinSavePopover] = useState(false);

  const runnerUIRef = useRef<RunnerUIWrapperRef>(null);

  const { toast } = useToast();

  const TUTORIAL_STORAGE_KEY = "shepherd-tour";

  useEffect(() => {
    if (params.get("resetTutorial") === "true") {
      localStorage.removeItem(TUTORIAL_STORAGE_KEY);
      router.push(pathname);
    } else if (!localStorage.getItem(TUTORIAL_STORAGE_KEY)) {
      startTutorial(setPinBlocksPopover, setPinSavePopover);
      localStorage.setItem(TUTORIAL_STORAGE_KEY, "yes");
    }
  }, [availableNodes, router, pathname, params]);

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

  const onNodeDragStart = (_: MouseEvent, node: Node) => {
    initialPositionRef.current[node.id] = { ...node.position };
    isDragging.current = true;
  };

  const onNodeDragEnd = (_: MouseEvent, node: Node | null) => {
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
        undo: () => updateNode(node.id, { position: oldPosition }),
        redo: () => updateNode(node.id, { position: newPosition }),
      });
    }
    delete initialPositionRef.current[node.id];
  };

  // Function to clear status, output, and close the output info dropdown of all nodes
  // and reset data beads on edges
  const clearNodesStatusAndOutput = useCallback(() => {
    setNodes((nds) => {
      const newNodes = nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
          status: undefined,
          isOutputOpen: false,
        },
      }));

      return newNodes;
    });
  }, [setNodes]);

  const onNodesChange = useCallback(
    (nodeChanges: NodeChange<CustomNode>[]) => {
      // Persist the changes
      setNodes((prev) => applyNodeChanges(nodeChanges, prev));

      // Remove all edges that were connected to deleted nodes
      nodeChanges
        .filter((change) => change.type === "remove")
        .forEach((deletedNode) => {
          const nodeID = deletedNode.id;
          const deletedNodeData = nodes.find((node) => node.id === nodeID);

          if (deletedNodeData) {
            history.push({
              type: "DELETE_NODE",
              payload: { node: deletedNodeData },
              undo: () => addNodes(deletedNodeData),
              redo: () => deleteElements({ nodes: [{ id: nodeID }] }),
            });
          }

          const connectedEdges = edges.filter((edge) =>
            [edge.source, edge.target].includes(nodeID),
          );
          deleteElements({
            edges: connectedEdges.map((edge) => ({ id: edge.id })),
          });
        });
    },
    [deleteElements, setNodes, nodes, edges, addNodes],
  );

  const formatEdgeID = useCallback((conn: Link | Connection): string => {
    if ("sink_id" in conn) {
      return `${conn.source_id}_${conn.source_name}_${conn.sink_id}_${conn.sink_name}`;
    } else {
      return `${conn.source}_${conn.sourceHandle}_${conn.target}_${conn.targetHandle}`;
    }
  }, []);

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      // Check if this exact connection already exists
      const existingConnection = edges.find(
        (edge) =>
          edge.source === connection.source &&
          edge.target === connection.target &&
          edge.sourceHandle === connection.sourceHandle &&
          edge.targetHandle === connection.targetHandle,
      );

      if (existingConnection) {
        console.warn("This exact connection already exists.");
        return;
      }

      const edgeColor = getTypeColor(
        getOutputType(nodes, connection.source!, connection.sourceHandle!),
      );
      const sourceNode = getNode(connection.source!);
      const newEdge: CustomEdge = {
        id: formatEdgeID(connection),
        type: "custom",
        markerEnd: {
          type: MarkerType.ArrowClosed,
          strokeWidth: 2,
          color: edgeColor,
        },
        data: {
          edgeColor,
          sourcePos: sourceNode!.position,
          isStatic: sourceNode!.data.isOutputStatic,
        },
        ...connection,
        source: connection.source!,
        target: connection.target!,
      };

      addEdges(newEdge);
      history.push({
        type: "ADD_EDGE",
        payload: { edge: newEdge },
        undo: () => {
          deleteElements({ edges: [{ id: newEdge.id }] });
        },
        redo: () => {
          addEdges(newEdge);
        },
      });
      clearNodesStatusAndOutput(); // Clear status and output on connection change
    },
    [
      getNode,
      addEdges,
      deleteElements,
      clearNodesStatusAndOutput,
      nodes,
      edges,
      formatEdgeID,
      getOutputType,
    ],
  );

  const onEdgesChange = useCallback(
    (edgeChanges: EdgeChange<CustomEdge>[]) => {
      // Persist the changes
      setEdges((prev) => applyEdgeChanges(edgeChanges, prev));

      // Propagate edge changes to node data
      const addedEdges = edgeChanges.filter((change) => change.type === "add"),
        replaceEdges = edgeChanges.filter(
          (change) => change.type === "replace",
        ),
        removedEdges = edgeChanges.filter((change) => change.type === "remove"),
        selectedEdges = edgeChanges.filter(
          (change) => change.type === "select",
        );

      if (addedEdges.length > 0 || removedEdges.length > 0) {
        setNodes((nds) => {
          const newNodes = nds.map((node) => ({
            ...node,
            data: {
              ...node.data,
              connections: [
                // Remove node connections for deleted edges
                ...node.data.connections.filter(
                  (conn) =>
                    !removedEdges.some(
                      (removedEdge) => removedEdge.id === conn.edge_id,
                    ),
                ),
                // Add node connections for added edges
                ...addedEdges.map((addedEdge) => ({
                  edge_id: addedEdge.item.id,
                  source: addedEdge.item.source,
                  target: addedEdge.item.target,
                  sourceHandle: addedEdge.item.sourceHandle!,
                  targetHandle: addedEdge.item.targetHandle!,
                })),
              ],
            },
          }));

          return newNodes;
        });

        if (removedEdges.length > 0) {
          clearNodesStatusAndOutput(); // Clear status and output on edge deletion
        }
      }

      if (replaceEdges.length > 0) {
        // Reset node connections for all edges
        console.warn(
          "useReactFlow().setRootEdges was used to overwrite all edges. " +
            "Use addEdges, deleteElements, or reconnectEdge for incremental changes.",
          replaceEdges,
        );
        setNodes((nds) =>
          nds.map((node) => ({
            ...node,
            data: {
              ...node.data,
              connections: [
                ...replaceEdges.map((replaceEdge) => ({
                  edge_id: replaceEdge.item.id,
                  source: replaceEdge.item.source,
                  target: replaceEdge.item.target,
                  sourceHandle: replaceEdge.item.sourceHandle!,
                  targetHandle: replaceEdge.item.targetHandle!,
                })),
              ],
            },
          })),
        );
        clearNodesStatusAndOutput();
      }
    },
    [setNodes, clearNodesStatusAndOutput, setEdges],
  );

  const getNextNodeId = useCallback(() => {
    return uuidv4();
  }, []);

  const { x, y, zoom } = useViewport();

  const addNode = useCallback(
    (blockId: string, nodeType: string) => {
      const nodeSchema = availableNodes.find((node) => node.id === blockId);
      if (!nodeSchema) {
        console.error(`Schema not found for block ID: ${blockId}`);
        return;
      }

      // Calculate the center of the viewport considering zoom
      const viewportCenter = {
        x: (window.innerWidth / 2 - x) / zoom,
        y: (window.innerHeight / 2 - y) / zoom,
      };

      const newNode: CustomNode = {
        id: nodeId.toString(),
        type: "custom",
        position: viewportCenter, // Set the position to the calculated viewport center
        data: {
          blockType: nodeType,
          blockCosts: nodeSchema.costs,
          title: `${nodeType} ${nodeId}`,
          description: nodeSchema.description,
          categories: nodeSchema.categories,
          inputSchema: nodeSchema.inputSchema,
          outputSchema: nodeSchema.outputSchema,
          hardcodedValues: {},
          connections: [],
          isOutputOpen: false,
          block_id: blockId,
          isOutputStatic: nodeSchema.staticOutput,
          uiType: nodeSchema.uiType,
        },
      };

      addNodes(newNode);
      setNodeId((prevId) => prevId + 1);
      clearNodesStatusAndOutput(); // Clear status and output when a new node is added

      history.push({
        type: "ADD_NODE",
        payload: { node: newNode.data },
        undo: () => deleteElements({ nodes: [{ id: newNode.id }] }),
        redo: () => addNodes(newNode),
      });
    },
    [
      nodeId,
      availableNodes,
      addNodes,
      deleteElements,
      clearNodesStatusAndOutput,
      x,
      y,
      zoom,
    ],
  );

  const handleUndo = () => {
    history.undo();
  };

  const handleRedo = () => {
    history.redo();
  };

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Prevent copy/paste if any modal is open or if the focus is on an input element
      const activeElement = document.activeElement;
      const isInputField =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.getAttribute("contenteditable") === "true";

      if (isAnyModalOpen || isInputField) return;

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
            const oldToNewNodeIDMap: Record<string, string> = {};
            const pastedNodes = copiedNodes.map((node, index) => {
              const newNodeId = (nodeId + index).toString();
              oldToNewNodeIDMap[node.id] = newNodeId;
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
                  executionResults: undefined, // Clear output data
                },
              };
            });
            setNodes((existingNodes) =>
              // Deselect copied nodes
              existingNodes.map((node) => ({ ...node, selected: false })),
            );
            addNodes(pastedNodes);
            setNodeId((prevId) => prevId + copiedNodes.length);

            const pastedEdges = copiedEdges.map((edge) => {
              const newSourceId = oldToNewNodeIDMap[edge.source] ?? edge.source;
              const newTargetId = oldToNewNodeIDMap[edge.target] ?? edge.target;
              return {
                ...edge,
                id: `${newSourceId}_${edge.sourceHandle}_${newTargetId}_${edge.targetHandle}_${Date.now()}`,
                source: newSourceId,
                target: newTargetId,
              };
            });
            addEdges(pastedEdges);
          }
        }
      }
    },
    [
      isAnyModalOpen,
      nodes,
      edges,
      copiedNodes,
      setNodes,
      addNodes,
      copiedEdges,
      addEdges,
      nodeId,
    ],
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
      icon: <IconUndo2 />,
      onClick: handleUndo,
    },
    {
      label: "Redo",
      icon: <IconRedo2 />,
      onClick: handleRedo,
    },
  ];

  return (
    <FlowContext.Provider
      value={{ visualizeBeads, setIsAnyModalOpen, getNextNodeId }}
    >
      <div className={className}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={{ custom: CustomNode }}
          edgeTypes={{ custom: CustomEdge }}
          connectionLineComponent={ConnectionLine}
          onConnect={onConnect}
          onNodesChange={onNodesChange}
          onNodesDelete={onNodesDelete}
          onEdgesChange={onEdgesChange}
          onNodeDragStop={onNodeDragEnd}
          onNodeDragStart={onNodeDragStart}
          deleteKeyCode={["Backspace", "Delete"]}
          minZoom={0.2}
          maxZoom={2}
        >
          <Controls />
          <Background />
          <ControlPanel
            className="absolute z-10"
            controls={editorControls}
            topChildren={
              <BlocksControl
                pinBlocksPopover={pinBlocksPopover} // Pass the state to BlocksControl
                blocks={availableNodes}
                addBlock={addNode}
              />
            }
            botChildren={
              <SaveControl
                agentMeta={savedAgent}
                onSave={(isTemplate) => requestSave(isTemplate ?? false)}
                agentDescription={agentDescription}
                onDescriptionChange={setAgentDescription}
                agentName={agentName}
                onNameChange={setAgentName}
                pinSavePopover={pinSavePopover}
              />
            }
          ></ControlPanel>
          <PrimaryActionBar
            onClickAgentOutputs={() => runnerUIRef.current?.openRunnerOutput()}
            onClickRunAgent={() => {
              if (!savedAgent) {
                toast({
                  title: `Please save the agent using the button in the left sidebar before running it.`,
                  duration: 2000,
                });
                return;
              }
              if (!isRunning) {
                runnerUIRef.current?.runOrOpenInput();
              } else {
                requestStopRun();
              }
            }}
            isDisabled={!savedAgent}
            isRunning={isRunning}
            requestStopRun={requestStopRun}
            runAgentTooltip={!isRunning ? "Run Agent" : "Stop Agent"}
          />
        </ReactFlow>
      </div>
      <RunnerUIWrapper
        ref={runnerUIRef}
        nodes={nodes}
        setNodes={setNodes}
        isRunning={isRunning}
        requestSaveAndRun={requestSaveAndRun}
      />
    </FlowContext.Provider>
  );
};

const WrappedFlowEditor: typeof FlowEditor = (props) => (
  <ReactFlowProvider>
    <FlowEditor {...props} />
  </ReactFlowProvider>
);

export default WrappedFlowEditor;
