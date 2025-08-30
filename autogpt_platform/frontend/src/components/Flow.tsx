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
  OnConnectStart,
  OnConnectEnd,
  Connection,
  MarkerType,
  NodeChange,
  EdgeChange,
  useReactFlow,
  applyEdgeChanges,
  applyNodeChanges,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { CustomNode } from "./CustomNode";
import "./flow.css";
import {
  BlockUIType,
  formatEdgeID,
  GraphExecutionID,
  GraphID,
  LibraryAgent,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { Key, storage } from "@/services/storage/local-storage";
import {
  getTypeColor,
  findNewlyAddedBlockCoordinates,
  beautifyString,
} from "@/lib/utils";
import { history } from "./history";
import { CustomEdge } from "./CustomEdge";
import ConnectionLine from "./ConnectionLine";
import { Control, ControlPanel } from "@/components/edit/control/ControlPanel";
import { SaveControl } from "@/components/edit/control/SaveControl";
import { BlocksControl } from "@/components/edit/control/BlocksControl";
import { FloatingBlocksMenu } from "@/components/edit/control/FloatingBlocksMenu";
import {
  ConnectionSelector,
  DynamicKeyDialog,
} from "@/components/edit/control/ConnectionSelector";
import { DictConnectionDialog } from "@/components/edit/control/DictConnectionDialog";
import {
  getCompatibleInputs,
  getCompatibleOutputs,
  getAdditionalPropertiesType,
  getArrayItemsType,
} from "@/lib/utils/connectionUtils";
import { IconUndo2, IconRedo2 } from "@/components/ui/icons";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { startTutorial } from "./tutorial";
import useAgentGraph from "@/hooks/useAgentGraph";
import { v4 as uuidv4 } from "uuid";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import RunnerUIWrapper, {
  RunnerUIWrapperRef,
} from "@/components/RunnerUIWrapper";
import PrimaryActionBar from "@/components/PrimaryActionButton";
import OttoChatWidget from "@/components/OttoChatWidget";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useCopyPaste } from "../hooks/useCopyPaste";
import NewControlPanel from "@/app/(platform)/build/components/NewBlockMenu/NewControlPanel/NewControlPanel";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

// This is for the history, this is the minimum distance a block must move before it is logged
// It helps to prevent spamming the history with small movements especially when pressing on a input in a block
const MINIMUM_MOVE_BEFORE_LOG = 50;

type FlowContextType = {
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

export const FlowContext = createContext<FlowContextType | null>(null);

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
    updateNodeData,
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

  // Connection drag state
  const [connectionDragState, setConnectionDragState] = useState<{
    sourceNodeId: string | null;
    sourceHandle: string | null;
    sourceHandleType: "source" | "target" | null;
    dropPosition: { x: number; y: number } | null;
  }>({
    sourceNodeId: null,
    sourceHandle: null,
    sourceHandleType: null,
    dropPosition: null,
  });
  const [showFloatingMenu, setShowFloatingMenu] = useState(false);
  const isSelectingBlockRef = useRef(false); // Track if we're in the process of selecting a block

  // Use a ref to track connection state immediately (state updates are async)
  const connectionDragRef = useRef<{
    sourceNodeId: string | null;
    sourceHandle: string | null;
    sourceHandleType: "source" | "target" | null;
  }>({
    sourceNodeId: null,
    sourceHandle: null,
    sourceHandleType: null,
  });
  const [connectionSelectorState, setConnectionSelectorState] = useState<{
    isOpen: boolean;
    newNodeId: string | null;
    options: Array<{
      handleId: string;
      schema: any;
      isRequired?: boolean;
      allowDynamicKey?: boolean;
      dynamicKeyType?: string;
      isDynamic?: boolean;
    }>;
    title: string;
    allowDynamicKey: boolean;
    dynamicKeyType?: string;
  }>({
    isOpen: false,
    newNodeId: null,
    options: [],
    title: "",
    allowDynamicKey: false,
  });
  const [dynamicKeyDialogState, setDynamicKeyDialogState] = useState<{
    isOpen: boolean;
    nodeId: string | null;
    handleId: string | null;
    isArray: boolean;
    keyType?: string;
  }>({
    isOpen: false,
    nodeId: null,
    handleId: null,
    isArray: false,
  });

  const [dictConnectionState, setDictConnectionState] = useState<{
    isOpen: boolean;
    newNodeId: string | null;
    handleId: string | null;
    keyType: string;
    valueType: string;
    sourceType: string;
  }>({
    isOpen: false,
    newNodeId: null,
    handleId: null,
    keyType: "string",
    valueType: "any",
    sourceType: "any",
  });

  const {
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
  const api = useBackendAPI();
  const [libraryAgent, setLibraryAgent] = useState<LibraryAgent | null>(null);
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
        console.warn(
          "[onConnect] ❌ This exact connection already exists:",
          existingConnection,
        );
        return;
      }

      const outputType = getOutputType(
        nodes,
        connection.source!,
        connection.sourceHandle!,
      );

      const edgeColor = getTypeColor(outputType);
      const sourceNode = getNode(connection.source!);
      const targetNode = getNode(connection.target!);

      // Check if both nodes exist
      if (!sourceNode || !targetNode) {
        console.error("[onConnect] ❌ Source or target node not found:", {
          sourceNode: sourceNode?.id,
          targetNode: targetNode?.id,
          connection,
        });
        return;
      }

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
          sourcePos: sourceNode.position,
          isStatic: sourceNode.data.isOutputStatic,
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

  const onConnectStart: OnConnectStart = useCallback((event, params) => {
    // Track the source of the connection drag
    if (params?.nodeId && params?.handleId && params?.handleType) {
      // Update ref immediately
      connectionDragRef.current = {
        sourceNodeId: params.nodeId,
        sourceHandle: params.handleId,
        sourceHandleType: params.handleType,
      };

      // Also update state for other components
      setConnectionDragState({
        sourceNodeId: params.nodeId,
        sourceHandle: params.handleId,
        sourceHandleType: params.handleType,
        dropPosition: null,
      });
    }
  }, []);

  const onConnectEnd: OnConnectEnd = useCallback(
    (event) => {
      // Use the ref to check if we were dragging a connection
      const isDraggingConnection =
        connectionDragRef.current.sourceNodeId !== null;

      if (!isDraggingConnection) {
        // Not dragging a connection, ignore
        return;
      }

      // Check if the connection was dropped without connecting
      const target = event.target as HTMLElement;

      // Check if the target is the pane or its child elements
      const isPane =
        target.classList.contains("react-flow__pane") ||
        target.classList.contains("react-flow__background") ||
        target.closest(".react-flow__pane");

      // Also check if we're not over a handle or node
      const isOverHandle = target.closest(".react-flow__handle");
      const isOverNode = target.closest(".react-flow__node");

      if (isPane && !isOverHandle && !isOverNode) {
        // Get the drop position in screen coordinates (for the menu)
        const mouseEvent = event as unknown as MouseEvent;
        const screenPosition = {
          x: mouseEvent.clientX,
          y: mouseEvent.clientY,
        };

        // Update state to show the floating menu
        setConnectionDragState({
          ...connectionDragRef.current,
          dropPosition: screenPosition, // Use screen coordinates for menu positioning
        });
        setShowFloatingMenu(true);
        // DON'T reset the ref here - we need it for the floating menu selection
      } else {
        // Reset state if connected or cancelled
        setConnectionDragState({
          sourceNodeId: null,
          sourceHandle: null,
          sourceHandleType: null,
          dropPosition: null,
        });
        setShowFloatingMenu(false);

        // Reset the ref only when not showing the floating menu
        connectionDragRef.current = {
          sourceNodeId: null,
          sourceHandle: null,
          sourceHandleType: null,
        };
      }
    },
    [screenToFlowPosition],
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

  const addNode = useCallback(
    (
      blockId: string,
      nodeType: string,
      hardcodedValues: any = {},
      position?: { x: number; y: number },
    ) => {
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
        position ||
        (nodeDimensions && Object.keys(nodeDimensions).length > 0
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
            });

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

      return newNode.id; // Return the new node ID for connection handling
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

  const handleBlockSelectFromFloatingMenu = useCallback(
    (
      blockId: string,
      blockName: string,
      hardcodedValues: Record<string, any>,
    ) => {
      // Mark that we're selecting a block
      isSelectingBlockRef.current = true;

      // Close the floating menu immediately
      setShowFloatingMenu(false);
      // Convert screen position back to flow coordinates for node placement
      const flowPosition = connectionDragState.dropPosition
        ? screenToFlowPosition({
            x: connectionDragState.dropPosition.x,
            y: connectionDragState.dropPosition.y,
          })
        : undefined;

      // Add the new node at the drop position
      const newNodeId = addNode(
        blockId,
        blockName,
        hardcodedValues,
        flowPosition,
      );

      if (
        !newNodeId ||
        !connectionDragState.sourceNodeId ||
        !connectionDragState.sourceHandle
      ) {
        setShowFloatingMenu(false);
        setConnectionDragState({
          sourceNodeId: null,
          sourceHandle: null,
          sourceHandleType: null,
          dropPosition: null,
        });
        return;
      }

      // Get the new node's schema to check compatible connections
      const newNodeSchema = availableBlocks.find((b) => b.id === blockId);
      if (!newNodeSchema) {
        setShowFloatingMenu(false);
        return;
      }

      // Determine compatible connections
      const sourceNode = nodes.find(
        (n) => n.id === connectionDragState.sourceNodeId,
      );
      if (!sourceNode) {
        setShowFloatingMenu(false);
        return;
      }

      let compatibleConnections: Array<{
        handleId: string;
        schema: any;
        isRequired?: boolean;
        isDynamic?: boolean;
        needsUserChoice?: boolean;
      }> = [];
      let connectionDirection: "source-to-target" | "target-to-source";

      if (connectionDragState.sourceHandleType === "source") {
        // Dragging from output to input
        const sourceSchema =
          sourceNode.data.outputSchema?.properties?.[
            connectionDragState.sourceHandle
          ];
        const sourceType = sourceSchema?.type;

        // Include dynamic connections (dict/array)
        compatibleConnections = getCompatibleInputs(
          newNodeSchema,
          sourceType,
          true,
        )
          .filter((inputKey) => {
            // Exclude credential fields from connection selector
            const inputSchema = newNodeSchema.inputSchema.properties[inputKey];
            return !("credentials_provider" in inputSchema);
          })
          .map((inputKey) => {
            const inputSchema = newNodeSchema.inputSchema.properties[inputKey];
            // Only mark as dynamic if:
            // 1. Target is array AND source is NOT array (and source is not 'any' or undefined)
            // 2. Target is dict AND source is NOT dict (and source is not 'any' or undefined)
            const isDynamic =
              (inputSchema.type === "array" &&
                sourceType !== "array" &&
                (sourceType as string) !== "any" &&
                sourceType !== undefined) ||
              (inputSchema.type === "object" &&
                "additionalProperties" in inputSchema &&
                inputSchema.additionalProperties &&
                sourceType !== "object" &&
                (sourceType as string) !== "any" &&
                sourceType !== undefined);

            // Special case: if source is 'any' or undefined connecting to array/dict, we need user choice
            const needsUserChoice =
              ((sourceType as string) === "any" || sourceType === undefined) &&
              (inputSchema.type === "array" ||
                (inputSchema.type === "object" &&
                  "additionalProperties" in inputSchema &&
                  inputSchema.additionalProperties));

            return {
              handleId: inputKey,
              schema: inputSchema,
              isRequired:
                newNodeSchema.inputSchema.required?.includes(inputKey),
              isDynamic,
              needsUserChoice,
            };
          });
        connectionDirection = "source-to-target";
      } else {
        // Dragging from input to output
        const sourceSchema =
          sourceNode.data.inputSchema?.properties?.[
            connectionDragState.sourceHandle
          ];
        const sourceType = sourceSchema?.type;

        compatibleConnections = getCompatibleOutputs(newNodeSchema, sourceType)
          .filter((outputKey) => {
            // Exclude credential fields from connection selector
            const outputSchema =
              newNodeSchema.outputSchema.properties[outputKey];
            return !("credentials_provider" in outputSchema);
          })
          .map((outputKey) => ({
            handleId: outputKey,
            schema: newNodeSchema.outputSchema.properties[outputKey],
            isDynamic: false,
            needsUserChoice: false,
          }));
        connectionDirection = "target-to-source";
      }

      // Separate connections by type
      const directConnections = compatibleConnections.filter(
        (c) => !c.isDynamic && !c.needsUserChoice,
      );
      const dynamicConnections = compatibleConnections.filter(
        (c) => c.isDynamic,
      );
      const userChoiceConnections = compatibleConnections.filter(
        (c) => c.needsUserChoice,
      );

      if (compatibleConnections.length === 0) {
        // No compatible connections
        setShowFloatingMenu(false);
        toast({
          title: "No compatible connections",
          description:
            "The selected block has no compatible inputs/outputs for this connection.",
          variant: "destructive",
        });
      } else if (
        directConnections.length === 1 &&
        dynamicConnections.length === 0
      ) {
        // Single direct connection - auto-connect
        const connection: Connection =
          connectionDirection === "source-to-target"
            ? {
                source: connectionDragState.sourceNodeId,
                sourceHandle: connectionDragState.sourceHandle,
                target: newNodeId,
                targetHandle: directConnections[0].handleId,
              }
            : {
                source: newNodeId,
                sourceHandle: directConnections[0].handleId,
                target: connectionDragState.sourceNodeId,
                targetHandle: connectionDragState.sourceHandle,
              };

        // Delay the connection to ensure the new node is in the state
        setTimeout(() => {
          onConnect(connection);
        }, 50);

        setShowFloatingMenu(false);

        // Reset connection drag state after successful connection
        setConnectionDragState({
          sourceNodeId: null,
          sourceHandle: null,
          sourceHandleType: null,
          dropPosition: null,
        });
      } else if (
        userChoiceConnections.length > 0 &&
        directConnections.length === 0 &&
        dynamicConnections.length === 0
      ) {
        // Any type connecting to list/dict - need user choice
        const conn = userChoiceConnections[0];
        const isArray = conn.schema.type === "array";

        // Show connection selector with both direct and dynamic options
        setConnectionSelectorState({
          isOpen: true,
          newNodeId,
          options: [
            // Direct connection option
            {
              handleId: conn.handleId,
              schema: conn.schema,
              isRequired: conn.isRequired,
              allowDynamicKey: false,
            },
            // Dynamic connection option (if applicable)
            ...(isArray ||
            (conn.schema.type === "object" &&
              "additionalProperties" in conn.schema &&
              conn.schema.additionalProperties)
              ? [
                  {
                    handleId: conn.handleId,
                    schema: conn.schema,
                    isRequired: conn.isRequired,
                    allowDynamicKey: true,
                    dynamicKeyType: isArray
                      ? getArrayItemsType(conn.schema)
                      : getAdditionalPropertiesType(conn.schema),
                  },
                ]
              : []),
          ],
          title: isArray ? "Connect to Array" : "Connect to Dictionary",
          allowDynamicKey: false,
          dynamicKeyType: undefined,
        });
        setShowFloatingMenu(false);
      } else if (
        dynamicConnections.length === 1 &&
        directConnections.length === 0 &&
        userChoiceConnections.length === 0
      ) {
        // Single dynamic connection (non-list to list or non-dict to dict)
        const dynamicConn = dynamicConnections[0];
        const isArray = dynamicConn.schema.type === "array";

        if (isArray) {
          // For arrays, append to the base handle with [] notation
          // The backend will interpret this as an array append operation
          const connection: Connection =
            connectionDirection === "source-to-target"
              ? {
                  source: connectionDragState.sourceNodeId,
                  sourceHandle: connectionDragState.sourceHandle,
                  target: newNodeId,
                  targetHandle: dynamicConn.handleId, // Just use base handle, backend handles the append
                }
              : {
                  source: newNodeId,
                  sourceHandle: dynamicConn.handleId, // Just use base handle
                  target: connectionDragState.sourceNodeId,
                  targetHandle: connectionDragState.sourceHandle,
                };

          // Delay the connection to ensure the new node is in the state
          setTimeout(() => {
            onConnect(connection);
          }, 50);

          setShowFloatingMenu(false);

          // Reset connection drag state after successful connection
          setConnectionDragState({
            sourceNodeId: null,
            sourceHandle: null,
            sourceHandleType: null,
            dropPosition: null,
          });

          // Reset the ref as well
          connectionDragRef.current = {
            sourceNodeId: null,
            sourceHandle: null,
            sourceHandleType: null,
          };
        } else {
          // For dicts, show enhanced dialog for key-value connection
          const valueType = getAdditionalPropertiesType(dynamicConn.schema);
          const sourceSchema =
            sourceNode.data.outputSchema?.properties?.[
              connectionDragState.sourceHandle
            ];
          const sourceType = sourceSchema?.type || "any";

          setDictConnectionState({
            isOpen: true,
            newNodeId,
            handleId: dynamicConn.handleId,
            keyType: "string", // Dictionary keys are always strings
            valueType: valueType || "any",
            sourceType,
          });
          setShowFloatingMenu(false);
        }
      } else {
        // Multiple connections - show selector
        // For each connection, determine if it should allow dynamic keys
        const sourceSchema =
          connectionDragState.sourceHandleType === "source"
            ? sourceNode.data.outputSchema?.properties?.[
                connectionDragState.sourceHandle
              ]
            : sourceNode.data.inputSchema?.properties?.[
                connectionDragState.sourceHandle
              ];
        const sourceType = sourceSchema?.type;

        setConnectionSelectorState({
          isOpen: true,
          newNodeId,
          options: compatibleConnections.map((c) => {
            // Only allow dynamic key if:
            // 1. Target is array/dict AND source is not the same type (unless 'any' or undefined)
            // 2. For 'any' or undefined type, we provide both options
            const allowDynamic =
              c.isDynamic ||
              (((sourceType as string) === "any" || sourceType === undefined) &&
                (c.schema.type === "array" ||
                  (c.schema.type === "object" &&
                    "additionalProperties" in c.schema &&
                    c.schema.additionalProperties)));

            return {
              handleId: c.handleId,
              schema: c.schema,
              isRequired: c.isRequired,
              allowDynamicKey: allowDynamic,
              dynamicKeyType: allowDynamic
                ? c.schema.type === "array"
                  ? getArrayItemsType(c.schema)
                  : getAdditionalPropertiesType(c.schema)
                : undefined,
            };
          }),
          title: "Select Connection",
          allowDynamicKey: false, // This is now per-option
          dynamicKeyType: undefined,
        });
        setShowFloatingMenu(false);
        // Don't reset connection drag state yet - we need it for the selector
      }

      // Reset the selecting flag and connection ref after processing
      // Use a longer delay to ensure connections complete first
      setTimeout(() => {
        isSelectingBlockRef.current = false;
        // Reset the connection ref now that we're done with it
        connectionDragRef.current = {
          sourceNodeId: null,
          sourceHandle: null,
          sourceHandleType: null,
        };
      }, 200); // Increased delay to ensure connection completes
    },
    [
      addNode,
      connectionDragState,
      availableBlocks,
      nodes,
      onConnect,
      toast,
      screenToFlowPosition,
    ],
  );

  const handleDictConnection = useCallback(
    (key: string, _connectToValue: boolean, _staticValue?: string) => {
      if (
        !dictConnectionState.newNodeId ||
        !dictConnectionState.handleId ||
        !connectionDragState.sourceNodeId
      ) {
        return;
      }

      // For dict connections, we need to first update the node to create the dynamic handle
      const targetHandle = `${dictConnectionState.handleId}.${key}`;

      // Create connection with dict key appended to handle ID
      const connection: Connection =
        connectionDragState.sourceHandleType === "source"
          ? {
              source: connectionDragState.sourceNodeId,
              sourceHandle: connectionDragState.sourceHandle!,
              target: dictConnectionState.newNodeId,
              targetHandle: targetHandle,
            }
          : {
              source: dictConnectionState.newNodeId,
              sourceHandle: targetHandle,
              target: connectionDragState.sourceNodeId,
              targetHandle: connectionDragState.sourceHandle!,
            };

      // First, update the node's connections to trigger handle creation
      const targetNode = getNode(dictConnectionState.newNodeId);
      if (targetNode) {
        const newConnection = {
          edge_id: formatEdgeID(connection),
          source: connection.source!,
          sourceHandle: connection.sourceHandle!,
          target: connection.target!,
          targetHandle: connection.targetHandle!,
        };

        // Add the connection to the node's data to trigger handle creation
        updateNodeData(dictConnectionState.newNodeId, {
          connections: [...(targetNode.data.connections || []), newConnection],
        });

        // Then create the actual edge after a delay to ensure the handle exists
        setTimeout(() => {
          onConnect(connection);
        }, 100);
      } else {
        // Fallback if node not found
        setTimeout(() => {
          onConnect(connection);
        }, 50);
      }

      // Reset states
      setDictConnectionState({
        isOpen: false,
        newNodeId: null,
        handleId: null,
        keyType: "string",
        valueType: "any",
        sourceType: "any",
      });

      setConnectionDragState({
        sourceNodeId: null,
        sourceHandle: null,
        sourceHandleType: null,
        dropPosition: null,
      });

      connectionDragRef.current = {
        sourceNodeId: null,
        sourceHandle: null,
        sourceHandleType: null,
      };
    },
    [
      dictConnectionState,
      connectionDragState,
      onConnect,
      getNode,
      updateNodeData,
      formatEdgeID,
    ],
  );

  const handleConnectionSelect = useCallback(
    (handleId: string, dynamicKey?: string) => {
      if (
        !connectionSelectorState.newNodeId ||
        !connectionDragState.sourceNodeId
      ) {
        console.error("Missing node IDs:", {
          newNodeId: connectionSelectorState.newNodeId,
          sourceNodeId: connectionDragState.sourceNodeId,
        });
        return;
      }

      // Note: handleId might already have [] notation for arrays
      let targetHandle = handleId;

      // Handle dynamic connections
      let isDynamicArrayAppend = false;
      let arrayIndex = -1;
      if (dynamicKey) {
        if (dynamicKey === "__append__") {
          // Special marker for array append
          // We need to create a unique handle for this dynamic element
          isDynamicArrayAppend = true;

          // Get the target node to find the next index
          const targetNodeId =
            connectionDragState.sourceHandleType === "source"
              ? connectionSelectorState.newNodeId
              : connectionDragState.sourceNodeId;
          const targetNode = getNode(targetNodeId);

          if (targetNode) {
            const dynamicArrayIndices =
              targetNode.data.metadata?.dynamicArrayIndices || {};
            const currentIndices = dynamicArrayIndices[handleId] || [];
            arrayIndex = currentIndices.length;
            // We'll use indexed handle for the connection
            targetHandle = `${handleId}[${arrayIndex}]`;
          }
        } else if (!handleId.endsWith("[]")) {
          // Dict key - append with dot notation
          targetHandle = `${handleId}.${dynamicKey}`;
        }
      }

      const connection: Connection =
        connectionDragState.sourceHandleType === "source"
          ? {
              source: connectionDragState.sourceNodeId,
              sourceHandle: connectionDragState.sourceHandle!,
              target: connectionSelectorState.newNodeId,
              targetHandle,
            }
          : {
              source: connectionSelectorState.newNodeId,
              sourceHandle: targetHandle, // Use the modified handle for both source and target
              target: connectionDragState.sourceNodeId,
              targetHandle: connectionDragState.sourceHandle!,
            };

      // If this is an array append, we need to mark it and update the node first
      if (isDynamicArrayAppend && arrayIndex >= 0) {
        const targetNodeId =
          connectionDragState.sourceHandleType === "source"
            ? connectionSelectorState.newNodeId
            : connectionDragState.sourceNodeId;
        const targetNode = getNode(targetNodeId);

        if (targetNode) {
          // Add metadata to track this as a dynamic array append
          const dynamicArrayConnections =
            targetNode.data.metadata?.dynamicArrayConnections || {};
          const dynamicArrayIndices =
            targetNode.data.metadata?.dynamicArrayIndices || {};

          // Track this connection as dynamic
          dynamicArrayConnections[handleId] = true;
          const currentIndices = dynamicArrayIndices[handleId] || [];
          if (!currentIndices.includes(arrayIndex)) {
            currentIndices.push(arrayIndex);
          }
          dynamicArrayIndices[handleId] = currentIndices;

          // Update node metadata to trigger re-render with new handle
          // Store in metadata to avoid sending to backend
          updateNodeData(targetNodeId, {
            metadata: {
              ...targetNode.data.metadata,
              dynamicArrayConnections,
              dynamicArrayIndices,
            },
          });

          // Delay the connection to ensure the handle exists in the DOM
          setTimeout(() => {
            onConnect(connection);

            // Reset states after delayed connection
            setConnectionSelectorState({
              isOpen: false,
              newNodeId: null,
              options: [],
              title: "",
              allowDynamicKey: false,
              dynamicKeyType: undefined,
            });

            setConnectionDragState({
              sourceNodeId: null,
              sourceHandle: null,
              sourceHandleType: null,
              dropPosition: null,
            });

            connectionDragRef.current = {
              sourceNodeId: null,
              sourceHandle: null,
              sourceHandleType: null,
            };
          }, 100);

          // Don't create the connection immediately or reset states
          return;
        }
      }

      onConnect(connection);

      // Now reset both states
      setConnectionSelectorState({
        isOpen: false,
        newNodeId: null,
        options: [],
        title: "",
        allowDynamicKey: false,
      });

      // Reset connection drag state after connection is made
      setConnectionDragState({
        sourceNodeId: null,
        sourceHandle: null,
        sourceHandleType: null,
        dropPosition: null,
      });
      // Reset the ref as well
      connectionDragRef.current = {
        sourceNodeId: null,
        sourceHandle: null,
        sourceHandleType: null,
      };
    },
    [connectionSelectorState, connectionDragState, onConnect],
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
          onConnectStart={onConnectStart}
          onConnectEnd={onConnectEnd}
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
            />
          ) : (
            <ControlPanel
              className="absolute z-20"
              controls={editorControls}
              topChildren={
                <BlocksControl
                  pinBlocksPopover={pinBlocksPopover} // Pass the state to BlocksControl
                  blocks={availableBlocks}
                  addBlock={addNode}
                  flows={availableFlows}
                  nodes={nodes}
                />
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
                  pinSavePopover={pinSavePopover}
                />
              }
            />
          )}

          {!graphHasWebhookNodes ? (
            <PrimaryActionBar
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
                You can view its activity in your
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

        {/* Floating Blocks Menu */}
        {showFloatingMenu && connectionDragState.dropPosition && (
          <FloatingBlocksMenu
            blocks={availableBlocks}
            position={connectionDragState.dropPosition}
            onSelectBlock={handleBlockSelectFromFloatingMenu}
            onClose={() => {
              setShowFloatingMenu(false);
              // Only reset connection state if we're not in the process of selecting a block
              if (!isSelectingBlockRef.current) {
                setConnectionDragState({
                  sourceNodeId: null,
                  sourceHandle: null,
                  sourceHandleType: null,
                  dropPosition: null,
                });
                // Reset the ref when closing without selection
                connectionDragRef.current = {
                  sourceNodeId: null,
                  sourceHandle: null,
                  sourceHandleType: null,
                };
              }
            }}
            flows={availableFlows}
            nodes={nodes}
            connectionType={connectionDragState.sourceHandleType!}
            handleType={
              connectionDragState.sourceHandleType === "source"
                ? nodes.find((n) => n.id === connectionDragState.sourceNodeId)
                    ?.data.outputSchema?.properties?.[
                    connectionDragState.sourceHandle!
                  ]?.type
                : nodes.find((n) => n.id === connectionDragState.sourceNodeId)
                    ?.data.inputSchema?.properties?.[
                    connectionDragState.sourceHandle!
                  ]?.type
            }
            sourceNodeId={connectionDragState.sourceNodeId!}
            sourceHandle={connectionDragState.sourceHandle!}
          />
        )}

        {/* Connection Selector Dialog */}
        <ConnectionSelector
          isOpen={connectionSelectorState.isOpen}
          onClose={() => {
            // Reset selector state
            setConnectionSelectorState({
              isOpen: false,
              newNodeId: null,
              options: [],
              title: "",
              allowDynamicKey: false,
            });
            // Also reset connection drag state when dialog is cancelled
            setConnectionDragState({
              sourceNodeId: null,
              sourceHandle: null,
              sourceHandleType: null,
              dropPosition: null,
            });
            // Reset the ref as well
            connectionDragRef.current = {
              sourceNodeId: null,
              sourceHandle: null,
              sourceHandleType: null,
            };
          }}
          onSelect={handleConnectionSelect}
          options={connectionSelectorState.options}
          title={connectionSelectorState.title}
          description="Choose which handle to connect to"
          allowDynamicKey={connectionSelectorState.allowDynamicKey}
          dynamicKeyType={connectionSelectorState.dynamicKeyType}
        />

        {/* Dict Connection Dialog */}
        <DictConnectionDialog
          isOpen={dictConnectionState.isOpen}
          onClose={() => {
            setDictConnectionState({
              isOpen: false,
              newNodeId: null,
              handleId: null,
              keyType: "string",
              valueType: "any",
              sourceType: "any",
            });
            // Also reset connection state
            setConnectionDragState({
              sourceNodeId: null,
              sourceHandle: null,
              sourceHandleType: null,
              dropPosition: null,
            });
            connectionDragRef.current = {
              sourceNodeId: null,
              sourceHandle: null,
              sourceHandleType: null,
            };
          }}
          onConfirm={handleDictConnection}
          title="Add Dictionary Entry"
          description="Choose how to connect to the dictionary"
          keyType={dictConnectionState.keyType}
          valueType={dictConnectionState.valueType}
          sourceType={dictConnectionState.sourceType}
        />

        {/* Dynamic Key Dialog */}
        <DynamicKeyDialog
          isOpen={dynamicKeyDialogState.isOpen}
          onClose={() =>
            setDynamicKeyDialogState({
              isOpen: false,
              nodeId: null,
              handleId: null,
              isArray: false,
            })
          }
          onConfirm={(key) => {
            if (
              dynamicKeyDialogState.handleId &&
              dynamicKeyDialogState.nodeId
            ) {
              // Create connection with dynamic key
              // For arrays, append without index (backend handles appending)
              // For dicts, add the key with dot notation
              const targetHandle = dynamicKeyDialogState.isArray
                ? dynamicKeyDialogState.handleId // Arrays auto-append, no index needed
                : `${dynamicKeyDialogState.handleId}.${key}`; // Dict with key

              const connection: Connection =
                connectionDragState.sourceHandleType === "source"
                  ? {
                      source: connectionDragState.sourceNodeId!,
                      sourceHandle: connectionDragState.sourceHandle!,
                      target: dynamicKeyDialogState.nodeId,
                      targetHandle,
                    }
                  : {
                      source: dynamicKeyDialogState.nodeId,
                      sourceHandle: targetHandle,
                      target: connectionDragState.sourceNodeId!,
                      targetHandle: connectionDragState.sourceHandle!,
                    };

              onConnect(connection);

              // Reset states
              setConnectionDragState({
                sourceNodeId: null,
                sourceHandle: null,
                sourceHandleType: null,
                dropPosition: null,
              });
            }
            setDynamicKeyDialogState({
              isOpen: false,
              nodeId: null,
              handleId: null,
              isArray: false,
            });
          }}
          title={
            dynamicKeyDialogState.isArray
              ? "Add Array Index"
              : "Add Dictionary Key"
          }
          description={`Enter a ${dynamicKeyDialogState.isArray ? "index" : "key"} for the new connection`}
          keyType={dynamicKeyDialogState.keyType}
          isArray={dynamicKeyDialogState.isArray}
        />
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
    </FlowContext.Provider>
  );
};

const WrappedFlowEditor: typeof FlowEditor = (props) => (
  <ReactFlowProvider>
    <FlowEditor {...props} />
  </ReactFlowProvider>
);

export default WrappedFlowEditor;
