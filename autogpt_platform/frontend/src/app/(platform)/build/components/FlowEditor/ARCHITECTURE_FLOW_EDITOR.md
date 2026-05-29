# FlowEditor Architecture Documentation

## Overview

The FlowEditor is the core visual graph builder component of the AutoGPT Platform. It allows users to create, edit, and execute workflows by connecting nodes (blocks) together in a visual canvas powered by React Flow (XYFlow).

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Flow Component                        │
│  (Main container coordinating all sub-systems)              │
└───────────────┬──────────────────┬──────────────────────────┘
                │                  │
      ┌─────────▼────────┐  ┌─────▼──────────┐
      │   State Stores   │  │   React Flow   │
      │   (Zustand)      │  │   Canvas       │
      └────────┬─────────┘  └────────────────┘
               │
    ┌──────────┼──────────┬──────────┐
    │          │          │          │
┌───▼───┐  ┌──▼───┐  ┌───▼────┐  ┌─▼────────┐
│ Node  │  │ Edge │  │ Graph  │  │ Control  │
│ Store │  │ Store│  │ Store  │  │ Panel    │
└───────┘  └──────┘  └────────┘  └──────────┘
    │          │
    │          │
┌───▼──────────▼────────────────────────────────────┐
│            Custom Nodes & Edges                   │
│  (Visual components rendered on canvas)           │
└───────────────────────────────────────────────────┘
```

---

## Core Components Breakdown

### 1. **Flow Component** (`Flow/Flow.tsx`)

The main orchestrator component that brings everything together.

**Responsibilities:**

- Renders the ReactFlow canvas
- Integrates all stores (nodes, edges, graph state)
- Handles drag-and-drop for adding blocks
- Manages keyboard shortcuts (copy/paste)
- Controls lock state (editable vs read-only)

**Key Features:**

```tsx
<ReactFlow
  nodes={nodes}              // From nodeStore
  edges={edges}              // From edgeStore
  onNodesChange={...}        // Updates nodeStore
  onEdgesChange={...}        // Updates edgeStore
  onConnect={...}            // Creates new connections
  onDragOver={...}           // Enables block drag-drop
  onDrop={...}               // Adds blocks to canvas
/>
```

---

### 2. **State Management (Zustand Stores)**

The FlowEditor uses **4 primary Zustand stores** for state management:

#### **A. nodeStore** (`stores/nodeStore.ts`)

Manages all nodes (blocks) on the canvas.

**State:**

```typescript
{
  nodes: CustomNode[]                    // All nodes on canvas
  nodeCounter: number                    // Auto-increment for IDs
  nodeAdvancedStates: Record<string, boolean>  // Track advanced toggle
}
```

**Key Actions:**

- `addBlock()` - Creates a new block with position calculation
- `updateNodeData()` - Updates block's form values
- `addNodes()` - Bulk add (used when loading graph)
- `updateNodeStatus()` - Updates execution status (running/success/failed)
- `updateNodeExecutionResult()` - Stores output data from execution
- `getBackendNodes()` - Converts to backend format for saving

**Flow:**

1. User drags block from menu → `addBlock()` called
2. Block appears with unique ID at calculated position
3. User edits form → `updateNodeData()` updates hardcodedValues
4. On execution → status updates propagate via `updateNodeStatus()`

---

#### **B. edgeStore** (`stores/edgeStore.ts`)

Manages all connections (links) between nodes.

**State:**

```typescript
{
  edges: CustomEdge[]                    // All connections
  edgeBeads: Record<string, EdgeBead[]> // Animated data flow indicators
}
```

**Key Actions:**

- `addLinks()` - Creates connections between nodes
- `onConnect()` - Handles new connection creation
- `updateEdgeBeads()` - Shows animated data flow during execution
- `getBackendLinks()` - Converts to backend format

**Connection Logic:**

```
Source Node (output) → Edge → Target Node (input)
    └─ outputPin       │         └─ inputPin
                       │
                  (validated connection)
```

---

#### **C. graphStore** (`stores/graphStore.ts`)

Manages graph-level metadata and state.

**State:**

```typescript
{
  isGraphRunning: boolean                // Execution status
  inputSchema: Record<string, any>       // Graph-level inputs
  credentialsInputSchema: Record<...>    // Required credentials
  outputSchema: Record<string, any>      // Graph-level outputs
}
```

**Purpose:**

- Tracks if graph is currently executing
- Stores graph-level input/output schemas (for agent graphs)
- Used by BuilderActions to show/hide input/output panels

---

#### **D. controlPanelStore**

Manages UI state for the control panel (block menu, settings).

**State:**

```typescript
{
  blockMenuOpen: boolean;
  selectedBlock: BlockInfo | null;
}
```

---

### 3. **useFlow Hook** (`Flow/useFlow.ts`)

The main data-loading and initialization hook.

**Lifecycle:**

```
1. Component Mounts
   ↓
2. Read URL params (flowID, flowVersion, flowExecutionID)
   ↓
3. Fetch graph data from API
   ↓
4. Fetch block definitions for all blocks in graph
   ↓
5. Convert to CustomNodes
   ↓
6. Add nodes to nodeStore
   ↓
7. Add links to edgeStore
   ↓
8. If execution exists → fetch execution details
   ↓
9. Update node statuses and results
   ↓
10. Initialize history (undo/redo)
```

**Key Responsibilities:**

- **Data Fetching**: Loads graph, blocks, and execution data
- **Data Transformation**: Converts backend models to frontend CustomNodes
- **State Initialization**: Populates stores with loaded data
- **Drag & Drop**: Handles block drag-drop from menu
- **Cleanup**: Resets stores on unmount

**Important Effects:**

```typescript
// Load nodes when data is ready
useEffect(() => {
  if (customNodes.length > 0) {
    addNodes(customNodes);
  }
}, [customNodes]);

// Update node execution status in real-time
useEffect(() => {
  executionDetails.node_executions.forEach((nodeExecution) => {
    updateNodeStatus(nodeExecution.node_id, nodeExecution.status);
    updateNodeExecutionResult(nodeExecution.node_id, nodeExecution);
  });
}, [executionDetails]);
```

---

### 4. **Custom Nodes** (`nodes/CustomNode/`)

Visual representation of blocks on the canvas.

**Structure:**

```
CustomNode
├── NodeContainer (selection, context menu, positioning)
├── NodeHeader (title, icon, badges)
├── FormCreator (input fields using FormRenderer)
├── NodeAdvancedToggle (show/hide advanced fields)
├── OutputHandler (output connection points)
└── NodeDataRenderer (execution results display)
```

**Node Data Structure:**

```typescript
type CustomNodeData = {
  hardcodedValues: Record<string, any>; // User input values
  title: string; // Display name
  description: string; // Help text
  inputSchema: RJSFSchema; // Input form schema
  outputSchema: RJSFSchema; // Output schema
  uiType: BlockUIType; // UI variant (STANDARD, INPUT, OUTPUT, etc.)
  block_id: string; // Backend block ID
  status?: AgentExecutionStatus; // Execution state
  nodeExecutionResult?: NodeExecutionResult; // Output data
  costs: BlockCost[]; // Cost information
  categories: BlockInfoCategoriesItem[]; // Categorization
};
```

**Special Node Types:**

- `BlockUIType.NOTE` - Sticky note (no execution)
- `BlockUIType.INPUT` - Graph input (no left handles)
- `BlockUIType.OUTPUT` - Graph output (no right handles)
- `BlockUIType.WEBHOOK` - Webhook trigger
- `BlockUIType.AGENT` - Sub-agent execution

---

### 5. **Custom Edges** (`edges/CustomEdge.tsx`)

Visual connections between nodes with animated data flow.

**Features:**

- **Animated Beads**: Show data flowing during execution
- **Type-aware Styling**: Different colors for different data types
- **Validation**: Prevents invalid connections
- **Deletion**: Click to remove connection

**Bead Animation System:**

```
Node Execution Complete
    ↓
EdgeStore.updateEdgeBeads() called
    ↓
Beads created with output data
    ↓
CSS animation moves beads along edge path
    ↓
Beads removed after animation
```

---

### 6. **Handlers (Connection Points)** (`handlers/NodeHandle.tsx`)

The connection points on nodes where edges attach.

**Handle ID Format:**

```typescript
// Input handle: input-{propertyName}
"input-text_content";

// Output handle: output-{propertyName}
"output-result";
```

**Connection Validation:**

- Type compatibility checking
- Prevents cycles
- Single input connection enforcement
- Multiple output connections allowed

---

## Data Flow: Adding a Block

```
1. User drags block from BlockMenu
   ↓
2. onDragOver handler validates drop
   ↓
3. onDrop handler called
   ↓
4. Parse block data from dataTransfer
   ↓
5. Calculate position: screenToFlowPosition()
   ↓
6. nodeStore.addBlock(blockData, {}, position)
   ↓
7. New CustomNode created with:
   - Unique ID (nodeCounter++)
   - Initial position
   - Empty hardcodedValues
   - Block schema
   ↓
8. Node added to nodes array
   ↓
9. React Flow renders CustomNode component
   ↓
10. FormCreator renders input form
```

---

## Data Flow: Connecting Nodes

```
1. User drags from source handle to target handle
   ↓
2. React Flow calls onConnect()
   ↓
3. useCustomEdge hook processes:
   - Validate connection (type compatibility)
   - Generate edge ID
   - Check for cycles
   ↓
4. edgeStore.addEdge() creates CustomEdge
   ↓
5. Edge rendered on canvas
   ↓
6. Target node's input becomes "connected"
   ↓
7. FormRenderer hides input field (shows handle only)
```

---

## Data Flow: Graph Execution

```
1. User clicks "Run" in BuilderActions
   ↓
2. useSaveGraph hook saves current state
   ↓
3. API call: POST /execute
   ↓
4. Backend queues execution
   ↓
5. useFlowRealtime subscribes to WebSocket
   ↓
6. Execution updates stream in:
   - Node status changes (QUEUED → RUNNING → COMPLETED)
   - Node results
   ↓
7. useFlow updates:
   - nodeStore.updateNodeStatus()
   - nodeStore.updateNodeExecutionResult()
   - edgeStore.updateEdgeBeads() (animate data flow)
   ↓
8. UI reflects changes:
   - NodeExecutionBadge shows status
   - OutputHandler displays results
   - Edges animate with beads
```

---

## Data Flow: Saving a Graph

```
1. User edits form in CustomNode
   ↓
2. FormCreator calls handleChange()
   ↓
3. nodeStore.updateNodeData(nodeId, { hardcodedValues })
   ↓
4. historyStore.pushState() (for undo/redo)
   ↓
5. User clicks "Save"
   ↓
6. useSaveGraph hook:
   - nodeStore.getBackendNodes() → convert to backend format
   - edgeStore.getBackendLinks() → convert links
   ↓
7. API call: PUT /graph/:id
   ↓
8. Backend persists changes
```

---

## Key Utilities and Helpers

### **Position Calculation** (`components/helper.ts`)

```typescript
findFreePosition(existingNodes, width, margin);
// Finds empty space on canvas to place new block
// Uses grid-based collision detection
```

### **Node Conversion** (`components/helper.ts`)

```typescript
convertBlockInfoIntoCustomNodeData(blockInfo, hardcodedValues);
// Converts backend BlockInfo → CustomNodeData

convertNodesPlusBlockInfoIntoCustomNodes(node, blockInfo);
// Merges backend Node + BlockInfo → CustomNode (for loading)
```

### **Handle ID Generation** (`handlers/helpers.ts`)

```typescript
generateHandleId(fieldId);
// input-{fieldId} or output-{fieldId}
// Used to uniquely identify connection points
```

---

## Advanced Features

### **Copy/Paste** (`Flow/useCopyPaste.ts`)

- Duplicates selected nodes with offset positioning
- Preserves internal connections
- Does not copy external connections

### **Undo/Redo** (`stores/historyStore.ts`)

- Tracks state snapshots (nodes + edges)
- Maintains history stack
- Triggered on significant changes (add/remove/move)

### **Realtime Updates** (`Flow/useFlowRealtime.ts`)

- WebSocket connection for live execution updates
- Subscribes to execution events
- Updates node status and results in real-time

### **Advanced Fields Toggle**

- Each node tracks `showAdvanced` state
- Fields with `advanced: true` hidden by default
- Toggle button in node UI
- Connected fields always visible

---

## Integration Points

### **With Backend API**

```
GET /v1/graphs/:id         → Load graph
GET /v2/blocks             → Get block definitions
GET /v1/executions/:id     → Get execution details
PUT /v1/graphs/:id         → Save graph
POST /v1/graphs/:id/execute → Run graph
WebSocket /ws              → Real-time updates
```

### **With FormRenderer** (See ARCHITECTURE_INPUT_RENDERER.md)

```
CustomNode → FormCreator → FormRenderer
                           ↓
                    (RJSF-based form)
```

---

## Performance Considerations

1. **Memoization**: React.memo on CustomNode to prevent unnecessary re-renders
2. **Shallow Selectors**: useShallow() with Zustand to limit re-renders
3. **Lazy Loading**: Blocks fetched only when needed
4. **Debounced Saves**: Form changes debounced before triggering history
5. **Virtual Scrolling**: React Flow handles large graphs efficiently

---

## Common Patterns

### **Adding a New Block Type**

1. Define `BlockUIType` enum value
2. Create backend block with `uiType` field
3. Add conditional rendering in CustomNode if needed
4. Update handle visibility logic if required

### **Adding a New Field Type**

1. Create custom field in input-renderer/fields
2. Register in fields/index.ts
3. Use in block's inputSchema

### **Debugging Tips**

- Check browser DevTools → React Flow state
- Inspect Zustand stores: `useNodeStore.getState()`
- Look for console errors in edge validation
- Check WebSocket connection for realtime issues

---

## Common Issues & Solutions

**Issue**: Nodes not appearing after load

- **Check**: `customNodes` computed correctly in useFlow
- **Check**: `addNodes()` called after data fetched

**Issue**: Form not updating node data

- **Check**: `handleChange` in FormCreator wired correctly
- **Check**: `updateNodeData` called with correct nodeId

**Issue**: Edges not connecting

- **Check**: Handle IDs match between source and target
- **Check**: Type compatibility validation
- **Check**: No cycles created

**Issue**: Execution status not updating

- **Check**: WebSocket connection active
- **Check**: `flowExecutionID` in URL
- **Check**: `updateNodeStatus` called in useFlow effect

---

## Summary

The FlowEditor is a sophisticated visual workflow builder that:

1. Uses **React Flow** for canvas rendering
2. Manages state with **Zustand stores** (nodes, edges, graph, control)
3. Loads data via **useFlow hook** from backend API
4. Renders blocks as **CustomNodes** with dynamic forms
5. Connects blocks via **CustomEdges** with validation
6. Executes graphs with **real-time status updates**
7. Saves changes back to backend in structured format

The architecture prioritizes:

- **Separation of concerns** (stores, hooks, components)
- **Type safety** (TypeScript throughout)
- **Performance** (memoization, shallow selectors)
- **Developer experience** (clear data flow, utilities)
