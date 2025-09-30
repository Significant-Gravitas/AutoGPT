# FlowEditor Component

The FlowEditor is a powerful visual flow builder component built on top of React Flow that allows users to create, connect, and manage nodes in a visual workflow. It provides a comprehensive form system with dynamic input handling, connection management, and advanced features.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Store Management](#store-management)

## Architecture Overview

The FlowEditor follows a modular architecture with clear separation of concerns:

```
FlowEditor/
├── Flow.tsx                    # Main component
├── nodes/                      # Node-related components
│   ├── CustomNode.tsx         # Main node component
│   ├── FormCreator.tsx        # Dynamic form generator
│   ├── fields/                # Custom field components
│   ├── widgets/               # Custom input widgets
│   ├── templates/             # RJSF templates
│   └── helpers.ts             # Utility functions
├── edges/                      # Edge-related components
│   ├── CustomEdge.tsx         # Custom edge component
│   ├── useCustomEdge.ts       # Edge management hook
│   └── helpers.ts             # Edge utilities
├── handlers/                   # Connection handles
│   ├── NodeHandle.tsx         # Connection handle component
│   └── helpers.ts             # Handle utilities
├── components/                 # Shared components
│   ├── ArrayEditor/           # Array editing components
│   └── ObjectEditor/          # Object editing components
└── processors/                 # Data processors
    └── input-schema-pre-processor.ts
```

## Store Management

The FlowEditor uses Zustand for state management with two main stores:

### NodeStore (`useNodeStore`)

Manages all node-related state and operations.

**Key Features:**

- Node CRUD operations
- Advanced state management per node
- Form data persistence
- Node counter for unique IDs

**Usage:**

```typescript
import { useNodeStore } from "../stores/nodeStore";

// Get nodes
const nodes = useNodeStore(useShallow((state) => state.nodes));

// Add a new node
const addNode = useNodeStore((state) => state.addNode);

// Update node data
const updateNodeData = useNodeStore((state) => state.updateNodeData);

// Toggle advanced mode
const setShowAdvanced = useNodeStore((state) => state.setShowAdvanced);
```

**Store Methods:**

- `setNodes(nodes)` - Replace all nodes
- `addNode(node)` - Add a single node
- `addBlock(blockInfo)` - Add node from block info
- `updateNodeData(nodeId, data)` - Update node data
- `onNodesChange(changes)` - Handle node changes from React Flow
- `setShowAdvanced(nodeId, show)` - Toggle advanced mode
- `incrementNodeCounter()` - Get next node ID

### EdgeStore (`useEdgeStore`)

Manages all connection-related state and operations.

**Key Features:**

- Connection CRUD operations
- Connection validation
- Backend link conversion
- Connection state queries

**Usage:**

```typescript
import { useEdgeStore } from "../stores/edgeStore";

// Get connections
const connections = useEdgeStore((state) => state.connections);

// Add connection
const addConnection = useEdgeStore((state) => state.addConnection);

// Check if input is connected
const isInputConnected = useEdgeStore((state) => state.isInputConnected);
```

**Store Methods:**

- `setConnections(connections)` - Replace all connections
- `addConnection(conn)` - Add a new connection
- `removeConnection(edgeId)` - Remove connection by ID
- `upsertMany(conns)` - Bulk update connections
- `isInputConnected(nodeId, handle)` - Check input connection
- `isOutputConnected(nodeId, handle)` - Check output connection
- `getNodeConnections(nodeId)` - Get all connections for a node
- `getBackendLinks()` - Convert to backend format

## Form Creator System

The FormCreator is a dynamic form generator built on React JSON Schema Form (RJSF) that automatically creates forms based on JSON schemas.

### How It Works

1. **Schema Processing**: Input schemas are preprocessed to ensure all properties have types
2. **Widget Mapping**: Schema types are mapped to appropriate input widgets
3. **Field Rendering**: Custom fields handle complex data structures
4. **State Management**: Form data is automatically synced with the node store

### Key Components

#### FormCreator

```typescript
<FormCreator
  jsonSchema={preprocessedSchema}
  nodeId={nodeId}
/>
```

#### Custom Widgets

- `TextInputWidget` - Text, number, password inputs
- `SelectWidget` - Dropdown and multi-select
- `SwitchWidget` - Boolean toggles
- `FileWidget` - File upload
- `DateInputWidget` - Date picker
- `TimeInputWidget` - Time picker
- `DateTimeInputWidget` - DateTime picker

#### Custom Fields

- `AnyOfField` - Union type handling
- `ObjectField` - Free-form object editing
- `CredentialsField` - API credential management

#### Templates

- `FieldTemplate` - Custom field wrapper with handles
- `ArrayFieldTemplate` - Array editing interface
