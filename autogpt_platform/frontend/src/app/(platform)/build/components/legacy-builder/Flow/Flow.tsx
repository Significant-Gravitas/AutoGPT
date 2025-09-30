"use client";
import React, {
  createContext,
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  MouseEvent,
  Suspense,
} from "react";
import Link from "next/link";
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
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { CustomNode } from "../CustomNode/CustomNode";
import "./flow.css";
import {
  BlockUIType,
  formatEdgeID,
  GraphExecutionID,
  GraphID,
  LibraryAgent,
} from "@/lib/autogpt-server-api";
import { Key, storage } from "@/services/storage/local-storage";
import {
  getTypeColor,
  findNewlyAddedBlockCoordinates,
  beautifyString,
} from "@/lib/utils";
import { history } from "../history";
import { CustomEdge } from "../CustomEdge/CustomEdge";
import ConnectionLine from "../ConnectionLine";
import {
  Control,
  ControlPanel,
} from "@/app/(platform)/build/components/legacy-builder/ControlPanel";
import { SaveControl } from "@/app/(platform)/build/components/legacy-builder/SaveControl";
import { BlocksControl } from "@/app/(platform)/build/components/legacy-builder/BlocksControl";
import { GraphSearchControl } from "@/app/(platform)/build/components/legacy-builder/GraphSearchControl";
import { IconUndo2, IconRedo2 } from "@/components/__legacy__/ui/icons";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/molecules/Alert/Alert";
import { startTutorial } from "../tutorial";
import useAgentGraph from "@/hooks/useAgentGraph";
import { v4 as uuidv4 } from "uuid";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import RunnerUIWrapper, { RunnerUIWrapperRef } from "../RunnerUIWrapper";
import OttoChatWidget from "@/app/(platform)/build/components/legacy-builder/OttoChatWidget";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useCopyPaste } from "../../../../../../hooks/useCopyPaste";
import NewControlPanel from "@/app/(platform)/build/components/NewBlockMenu/NewControlPanel/NewControlPanel";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { BuildActionBar } from "../BuildActionBar";

// This is for the history, this is the minimum distance a block must move before it is logged
// It helps to prevent spamming the history with small movements especially when pressing on a input in a block
const MINIMUM_MOVE_BEFORE_LOG = 50;

type BuilderContextType = {
  libraryAgent: LibraryAgent | null;
  visualizeBeads: "no" | "static" | "animate";
  setIsAnyModalOpen: (isOpen: boolean) => void;
  getNextNodeId: () => string;
};

export type NodeDimension = {
  [nodeId: string]: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

export const BuilderContext = createContext<BuilderContextType | null>(null);

const FlowEditor: React.FC<{
  flowID?: GraphID;
  flowVersion?: number;
  className?: string;
}> = ({ flowID, flowVersion, className }) => {
  const {
    addNodes,
    addEdges,
    getNode,
    deleteElements,
    updateNode,
    getViewport,
    setViewport,
    screenToFlowPosition,
  } = useReactFlow<CustomNode, CustomEdge>();
  const [nodeId, setNodeId] = useState<number>(1);
  const [isAnyModalOpen, setIsAnyModalOpen] = useState(false);
  const [visualizeBeads] = useState<"no" | "static" | "animate">("animate");
  const [flowExecutionID, setFlowExecutionID] = useState<
    GraphExecutionID | undefined
  >();
  // State to control if blocks menu should be pinned open
  const [pinBlocksPopover, setPinBlocksPopover] = useState(false);
  // State to control if save popover should be pinned open
  const [pinSavePopover, setPinSavePopover] = useState(false);

  const {
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
    nodes,
    setNodes,
    edges,
    setEdges,
  } = useAgentGraph(
    flowID,
    flowVersion,
    flowExecutionID,
    visualizeBeads !== "no",
  );

  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const initialPositionRef = useRef<{
    [key: string]: { x: number; y: number };
  }>({});
  const isDragging = useRef(false);

  const runnerUIRef = useRef<RunnerUIWrapperRef>(null);

  const { toast } = useToast();

  // It stores the dimension of all nodes with position as well
  const [nodeDimensions, setNodeDimensions] = useState<NodeDimension>({});

  // Set page title with or without graph name
  useEffect(() => {
    document.title = savedAgent
      ? `${savedAgent.name} - Builder - AutoGPT Platform`
      : `Builder - AutoGPT Platform`;
  }, [savedAgent]);

  const graphHasWebhookNodes = useMemo(
    () =>
      nodes.some((n) =>
        [BlockUIType.WEBHOOK, BlockUIType.WEBHOOK_MANUAL].includes(
          n.data.uiType,
        ),
      ),
    [nodes],
  );

  useEffect(() => {
    if (params.get("resetTutorial") === "true") {
      storage.clean(Key.SHEPHERD_TOUR);
      router.push(pathname);
    } else if (!storage.get(Key.SHEPHERD_TOUR)) {
      const emptyNodes = (forceRemove: boolean = false) =>
        forceRemove ? (setNodes([]), setEdges([]), true) : nodes.length === 0;
      startTutorial(emptyNodes, setPinBlocksPopover, setPinSavePopover);
      storage.set(Key.SHEPHERD_TOUR, "yes");
    }
  }, [router, pathname, params, setEdges, setNodes, nodes.length]);

  useEffect(() => {
    if (params.get("open_scheduling") === "true") {
      runnerUIRef.current?.openRunInputDialog();
    }
    setFlowExecutionID(
      (params.get("flowExecutionID") as GraphExecutionID) || undefined,
    );
  }, [params]);

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
        history.undo();
      }

      if (isRedo) {
        event.preventDefault();
        history.redo();
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
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
          status: undefined,
          isOutputOpen: false,
        },
      })),
    );
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
              payload: { node: deletedNodeData.data },
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
          beadUp: 0,
          beadDown: 0,
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
        removedEdges = edgeChanges.filter((change) => change.type === "remove");

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

  // Set the initial view port to center the canvas.
  useEffect(() => {
    const { x, y } = getViewport();
    if (nodes.length <= 0 || x !== 0 || y !== 0) {
      return;
    }

    const topLeft = { x: Infinity, y: Infinity };
    const bottomRight = { x: -Infinity, y: -Infinity };

    nodes.forEach((node) => {
      const { x, y } = node.position;
      topLeft.x = Math.min(topLeft.x, x);
      topLeft.y = Math.min(topLeft.y, y);
      // Rough estimate of the width and height of the node: 500x400.
      bottomRight.x = Math.max(bottomRight.x, x + 500);
      bottomRight.y = Math.max(bottomRight.y, y + 400);
    });

    const centerX = (topLeft.x + bottomRight.x) / 2;
    const centerY = (topLeft.y + bottomRight.y) / 2;
    const zoom = 0.8;

    setViewport({
      x: window.innerWidth / 2 - centerX * zoom,
      y: window.innerHeight / 2 - centerY * zoom,
      zoom: zoom,
    });
  }, [nodes, getViewport, setViewport]);

  const navigateToNode = useCallback(
    (nodeId: string) => {
      const node = getNode(nodeId);
      if (!node) return;

      // Center the viewport on the selected node
      const zoom = 1.2; // Slightly zoom in for better visibility
      const nodeX = node.position.x + (node.width || 500) / 2;
      const nodeY = node.position.y + (node.height || 400) / 2;

      setViewport({
        x: window.innerWidth / 2 - nodeX * zoom,
        y: window.innerHeight / 2 - nodeY * zoom,
        zoom: zoom,
      });

      // Add a temporary highlight effect to the node
      updateNode(nodeId, {
        style: {
          ...node.style,
          boxShadow: "0 0 20px 5px rgba(59, 130, 246, 0.8)",
          transition: "box-shadow 0.3s ease-in-out",
        },
      });

      // Remove highlight after a delay
      setTimeout(() => {
        updateNode(nodeId, {
          style: {
            ...node.style,
            boxShadow: undefined,
          },
        });
      }, 2000);
    },
    [getNode, setViewport, updateNode],
  );

  const highlightNode = useCallback(
    (nodeId: string | null) => {
      if (!nodeId) {
        // Clear all highlights
        nodes.forEach((node) => {
          updateNode(node.id, {
            style: {
              ...node.style,
              boxShadow: undefined,
            },
          });
        });
        return;
      }

      const node = getNode(nodeId);
      if (!node) return;

      // Add highlight effect without moving view
      updateNode(nodeId, {
        style: {
          ...node.style,
          boxShadow: "0 0 15px 3px rgba(59, 130, 246, 0.6)",
          transition: "box-shadow 0.2s ease-in-out",
        },
      });
    },
    [getNode, updateNode, nodes],
  );

  const addNode = useCallback(
    (blockId: string, nodeType: string, hardcodedValues: any = {}) => {
      const nodeSchema = availableBlocks.find((node) => node.id === blockId);
      if (!nodeSchema) {
        console.error(`Schema not found for block ID: ${blockId}`);
        return;
      }

      /*
       Calculate a position to the right of the newly added block, allowing for some margin.
       If adding to the right side causes the new block to collide with an existing block, attempt to place it at the bottom or left.
       Why not the top? Because the height of the new block is unknown.
       If it still collides, run a loop to find the best position where it does not collide.
       Then, adjust the canvas to center on the newly added block.
       Note: The width is known, e.g., w = 300px for a note and w = 500px for others, but the height is dynamic.
       */

      // Alternative: We could also use D3 force, Intersection for this (React flow Pro examples)

      const { x, y } = getViewport();
      const viewportCoordinates =
        nodeDimensions && Object.keys(nodeDimensions).length > 0
          ? // we will get all the dimension of nodes, then store
            findNewlyAddedBlockCoordinates(
              nodeDimensions,
              nodeSchema.uiType == BlockUIType.NOTE ? 300 : 500,
              60,
              1.0,
            )
          : // we will get all the dimension of nodes, then store
            {
              x: window.innerWidth / 2 - x,
              y: window.innerHeight / 2 - y,
            };

      const newNode: CustomNode = {
        id: nodeId.toString(),
        type: "custom",
        position: viewportCoordinates, // Set the position to the calculated viewport center
        data: {
          blockType: nodeType,
          blockCosts: nodeSchema.costs,
          title: `${nodeType} ${nodeId}`,
          description: nodeSchema.description,
          categories: nodeSchema.categories,
          inputSchema: nodeSchema.inputSchema,
          outputSchema: nodeSchema.outputSchema,
          hardcodedValues: hardcodedValues,
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

      setViewport(
        {
          // Rough estimate of the dimension of the node is: 500x400px.
          // Though we skip shifting the X, considering the block menu side-bar.
          x: -viewportCoordinates.x * 0.8 + (window.innerWidth - 0.0) / 2,
          y: -viewportCoordinates.y * 0.8 + (window.innerHeight - 400) / 2,
          zoom: 0.8,
        },
        { duration: 500 },
      );

      history.push({
        type: "ADD_NODE",
        payload: { node: { ...newNode, ...newNode.data } },
        undo: () => deleteElements({ nodes: [{ id: newNode.id }] }),
        redo: () => addNodes(newNode),
      });
    },
    [
      nodeId,
      getViewport,
      setViewport,
      availableBlocks,
      addNodes,
      nodeDimensions,
      deleteElements,
      clearNodesStatusAndOutput,
    ],
  );

  const findNodeDimensions = useCallback(() => {
    const newNodeDimensions: NodeDimension = nodes.reduce((acc, node) => {
      const nodeElement = document.querySelector(
        `[data-id="custom-node-${node.id}"]`,
      );
      if (nodeElement) {
        const rect = nodeElement.getBoundingClientRect();
        const { left, top, width, height } = rect;

        const { x, y, zoom } = getViewport();

        // Convert screen coordinates to flow coordinates
        const flowX = (left - x) / zoom;
        const flowY = (top - y) / zoom;
        const flowWidth = width / zoom;
        const flowHeight = height / zoom;

        acc[node.id] = {
          x: flowX,
          y: flowY,
          width: flowWidth,
          height: flowHeight,
        };
      }
      return acc;
    }, {} as NodeDimension);

    setNodeDimensions(newNodeDimensions);
  }, [nodes, getViewport]);

  useEffect(() => {
    findNodeDimensions();
  }, [nodes, findNodeDimensions]);

  const handleCopyPaste = useCopyPaste(getNextNodeId);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Prevent copy/paste if any modal is open or if the focus is on an input element
      const activeElement = document.activeElement;
      const isInputField =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.getAttribute("contenteditable") === "true";

      if (isAnyModalOpen || isInputField) return;

      handleCopyPaste(event);
    },
    [isAnyModalOpen, handleCopyPaste],
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

  const editorControls: Control[] = useMemo(
    () => [
      {
        label: "Undo",
        icon: <IconUndo2 />,
        onClick: history.undo,
      },
      {
        label: "Redo",
        icon: <IconRedo2 />,
        onClick: history.redo,
      },
    ],
    [],
  );

  const handleRunButton = useCallback(async () => {
    if (isRunning) return;
    if (!savedAgent) {
      toast({
        title: `Please save the agent first, using the button in the left sidebar.`,
      });
      return;
    }
    await saveAgent();
    runnerUIRef.current?.runOrOpenInput();
  }, [isRunning, savedAgent, toast, saveAgent]);

  const handleScheduleButton = useCallback(async () => {
    if (isScheduling) return;
    if (!savedAgent) {
      toast({
        title: `Please save the agent first, using the button in the left sidebar.`,
      });
      return;
    }
    await saveAgent();
    runnerUIRef.current?.openRunInputDialog();
  }, [isScheduling, savedAgent, toast, saveAgent]);

  const isNewBlockEnabled = useGetFlag(Flag.NEW_BLOCK_MENU);
  const isGraphSearchEnabled = useGetFlag(Flag.GRAPH_SEARCH);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const blockData = event.dataTransfer.getData("application/reactflow");
      if (!blockData) return;

      try {
        const { blockId, blockName, hardcodedValues } = JSON.parse(blockData);

        // Convert screen coordinates to flow coordinates
        const position = screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        });

        // Find the block schema
        const nodeSchema = availableBlocks.find((node) => node.id === blockId);
        if (!nodeSchema) {
          console.error(`Schema not found for block ID: ${blockId}`);
          return;
        }

        // Create the new node at the drop position
        const newNode: CustomNode = {
          id: nodeId.toString(),
          type: "custom",
          position,
          data: {
            blockType: blockName,
            blockCosts: nodeSchema.costs || [],
            title: `${beautifyString(blockName)} ${nodeId}`,
            description: nodeSchema.description,
            categories: nodeSchema.categories,
            inputSchema: nodeSchema.inputSchema,
            outputSchema: nodeSchema.outputSchema,
            hardcodedValues: hardcodedValues,
            connections: [],
            isOutputOpen: false,
            block_id: blockId,
            uiType: nodeSchema.uiType,
          },
        };

        history.push({
          type: "ADD_NODE",
          payload: { node: { ...newNode, ...newNode.data } },
          undo: () => {
            deleteElements({ nodes: [{ id: newNode.id } as any], edges: [] });
          },
          redo: () => {
            addNodes([newNode]);
          },
        });
        addNodes([newNode]);
        clearNodesStatusAndOutput();

        setNodeId((prevId) => prevId + 1);
      } catch (error) {
        console.error("Failed to drop block:", error);
      }
    },
    [
      nodeId,
      availableBlocks,
      nodes,
      edges,
      addNodes,
      screenToFlowPosition,
      deleteElements,
      clearNodesStatusAndOutput,
    ],
  );

  return (
    <BuilderContext.Provider
      value={{ libraryAgent, visualizeBeads, setIsAnyModalOpen, getNextNodeId }}
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
          onDrop={onDrop}
          onDragOver={onDragOver}
          deleteKeyCode={["Backspace", "Delete"]}
          minZoom={0.1}
          maxZoom={2}
          className="dark:bg-slate-900"
        >
          <Controls />
          <Background className="dark:bg-slate-800" />
          {isNewBlockEnabled ? (
            <NewControlPanel
              flowExecutionID={flowExecutionID}
              visualizeBeads={visualizeBeads}
              pinSavePopover={pinSavePopover}
              pinBlocksPopover={pinBlocksPopover}
              // nodes={nodes}
              onNodeSelect={navigateToNode}
              onNodeHover={highlightNode}
            />
          ) : (
            <ControlPanel
              className="absolute z-20"
              controls={editorControls}
              topChildren={
                <>
                  <BlocksControl
                    pinBlocksPopover={pinBlocksPopover} // Pass the state to BlocksControl
                    blocks={availableBlocks}
                    addBlock={addNode}
                    flows={availableFlows}
                    nodes={nodes}
                  />
                  {isGraphSearchEnabled && (
                    <GraphSearchControl
                      nodes={nodes}
                      onNodeSelect={navigateToNode}
                      onNodeHover={highlightNode}
                    />
                  )}
                </>
              }
              botChildren={
                <SaveControl
                  agentMeta={savedAgent}
                  canSave={!isSaving && !isRunning && !isStopping}
                  onSave={saveAgent}
                  agentDescription={agentDescription}
                  onDescriptionChange={setAgentDescription}
                  agentName={agentName}
                  onNameChange={setAgentName}
                  agentRecommendedScheduleCron={agentRecommendedScheduleCron}
                  onRecommendedScheduleCronChange={
                    setAgentRecommendedScheduleCron
                  }
                  pinSavePopover={pinSavePopover}
                />
              }
            />
          )}

          {!graphHasWebhookNodes ? (
            <BuildActionBar
              className="absolute bottom-0 left-1/2 z-20 -translate-x-1/2"
              onClickAgentOutputs={runnerUIRef.current?.openRunnerOutput}
              onClickRunAgent={handleRunButton}
              onClickStopRun={stopRun}
              onClickScheduleButton={handleScheduleButton}
              isDisabled={!savedAgent}
              isRunning={isRunning}
            />
          ) : (
            <Alert className="absolute bottom-4 left-1/2 z-20 w-auto -translate-x-1/2 select-none">
              <AlertTitle>You are building a Trigger Agent</AlertTitle>
              <AlertDescription>
                Your agent{" "}
                {savedAgent?.nodes.some((node) => node.webhook)
                  ? "is listening"
                  : "will listen"}{" "}
                for its trigger and will run when the time is right.
                <br />
                You can view its activity in your{" "}
                <Link
                  href={
                    libraryAgent
                      ? `/library/agents/${libraryAgent.id}`
                      : "/library"
                  }
                  className="underline"
                >
                  Agent Library
                </Link>
                .
              </AlertDescription>
            </Alert>
          )}
        </ReactFlow>
      </div>
      {savedAgent && (
        <RunnerUIWrapper
          ref={runnerUIRef}
          graph={savedAgent}
          nodes={nodes}
          graphExecutionError={graphExecutionError}
          createRunSchedule={createRunSchedule}
          saveAndRun={saveAndRun}
        />
      )}
      <Suspense fallback={null}>
        <OttoChatWidget
          graphID={flowID}
          className="fixed bottom-4 right-4 z-20"
        />
      </Suspense>
    </BuilderContext.Provider>
  );
};

const WrappedFlowEditor: typeof FlowEditor = (props) => (
  <ReactFlowProvider>
    <FlowEditor {...props} />
  </ReactFlowProvider>
);

export default WrappedFlowEditor;
