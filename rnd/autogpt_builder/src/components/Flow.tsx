"use client";
import React, { useState, useCallback, useEffect, useMemo } from 'react';
import ReactFlow, {
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  Node,
  Edge,
  OnNodesChange,
  OnEdgesChange,
  OnConnect,
  NodeTypes,
  Connection,
} from 'reactflow';
import 'reactflow/dist/style.css';
import CustomNode from './CustomNode';
import './flow.css';
import AutoGPTServerAPI, { Block, Flow } from '@/lib/autogpt_server_api';
import { ObjectSchema } from '@/lib/types';

type CustomNodeData = {
  blockType: string;
  title: string;
  inputSchema: ObjectSchema;
  outputSchema: ObjectSchema;
  hardcodedValues: { [key: string]: any };
  setHardcodedValues: (values: { [key: string]: any }) => void;
  connections: Array<{ source: string; sourceHandle: string; target: string; targetHandle: string }>;
  isPropertiesOpen: boolean;
  status?: string;
  output_data?: any;
  block_id: string;
};

const Sidebar: React.FC<{isOpen: boolean, availableNodes: Block[], addNode: (id: string, name: string) => void}> =
  ({isOpen, availableNodes, addNode}) => {
  const [searchQuery, setSearchQuery] = useState('');

  if (!isOpen) return null;

  const filteredNodes = availableNodes.filter(node =>
    node.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className={`sidebar ${isOpen ? 'open' : ''}`}>
      <h3>Nodes</h3>
      <input
        type="text"
        placeholder="Search nodes..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
      />
      {filteredNodes.map((node) => (
        <div key={node.id} className="sidebarNodeRowStyle">
          <span>{node.name}</span>
          <button onClick={() => addNode(node.id, node.name)}>Add</button>
        </div>
      ))}
    </div>
  );
};

const FlowEditor: React.FC<{ flowID?: string }> = ({ flowID }) => {
  const [nodes, setNodes] = useState<Node<CustomNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [nodeId, setNodeId] = useState<number>(1);
  const [availableNodes, setAvailableNodes] = useState<Block[]>([]);
  const [agentId, setAgentId] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const apiUrl = 'http://localhost:8000';
  const api = new AutoGPTServerAPI(apiUrl);

  useEffect(() => {
    api.getBlocks()
      .then(blocks => setAvailableNodes(blocks))
      .catch();
  }, []);

  // Load existing flow
  useEffect(() => {
    if (!flowID || availableNodes.length == 0) return;

    api.getFlow(flowID)
      .then(flow => loadFlow(flow));
  }, [flowID, availableNodes]);

  const nodeTypes: NodeTypes = useMemo(() => ({ custom: CustomNode }), []);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge(connection, eds));
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === connection.target) {
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
                  } as { source: string; sourceHandle: string; target: string; targetHandle: string },
                ],
              },
            };
          }
          return node;
        })
      );
    },
    [setEdges, setNodes]
  );

  const addNode = (blockId: string, nodeType: string) => {
    const nodeSchema = availableNodes.find(node => node.id === blockId);
    if (!nodeSchema) {
      console.error(`Schema not found for block ID: ${blockId}`);
      return;
    }

    const newNode: Node<CustomNodeData> = {
      id: nodeId.toString(),
      type: 'custom',
      position: { x: Math.random() * 400, y: Math.random() * 400 },
      data: {
        blockType: nodeType,
        title: `${nodeType} ${nodeId}`,
        inputSchema: nodeSchema.inputSchema,
        outputSchema: nodeSchema.outputSchema,
        hardcodedValues: {},
        setHardcodedValues: (values: { [key: string]: any }) => {
          setNodes((nds) => nds.map((node) =>
            node.id === newNode.id
              ? { ...node, data: { ...node.data, hardcodedValues: values } }
              : node
          ));
        },
        connections: [],
        isPropertiesOpen: false,
        block_id: blockId,
      },
    };

    setNodes((nds) => [...nds, newNode]);
    setNodeId((prevId) => prevId + 1);
  };

  function loadFlow(flow: Flow) {
    setAgentId(flow.id);

    setNodes(flow.nodes.map(node => {
      const block = availableNodes.find(block => block.id === node.block_id)!;
      const newNode = {
        id: node.id,
        type: 'custom',
        position: { x: node.metadata.position.x, y: node.metadata.position.y },
        data: {
          block_id: block.id,
          blockType: block.name,
          title: `${block.name} ${node.id}`,
          inputSchema: block.inputSchema,
          outputSchema: block.outputSchema,
          hardcodedValues: {},
          setHardcodedValues: (values: { [key: string]: any; }) => {
            setNodes((nds) => nds.map((node) => node.id === newNode.id
              ? { ...node, data: { ...node.data, hardcodedValues: values } }
              : node
            ));
          },
          connections: [],
          isPropertiesOpen: false,
        },
      };
      return newNode;
    }));

    setEdges(flow.links.map(link => ({
      id: `${link.source_id}_${link.source_name}_${link.sink_id}_${link.sink_name}`,
      source: link.source_id,
      target: link.sink_id,
      sourceHandle: link.source_name || undefined,
      targetHandle: link.sink_name || undefined
    })));
  }

  const prepareNodeInputData = (node: Node<CustomNodeData>, allNodes: Node<CustomNodeData>[], allEdges: Edge[]) => {
    console.log("Preparing input data for node:", node.id, node.data.blockType);

    const blockSchema = availableNodes.find(n => n.id === node.data.block_id)?.inputSchema;

    if (!blockSchema) {
      console.error(`Schema not found for block ID: ${node.data.block_id}`);
      return {};
    }

    const getNestedData = (schema: ObjectSchema, values: { [key: string]: any }): { [key: string]: any } => {
      let inputData: { [key: string]: any } = {};

      if (schema.properties) {
        Object.keys(schema.properties).forEach((key) => {
          if (values[key] !== undefined) {
            if (schema.properties[key].type === 'object') {
              inputData[key] = getNestedData(schema.properties[key], values[key]);
            } else {
              inputData[key] = values[key];
            }
          }
        });
      }

      if (schema.additionalProperties) {
        inputData = { ...inputData, ...values };
      }

      return inputData;
    };

    let inputData = getNestedData(blockSchema, node.data.hardcodedValues);

    // Get data from connected nodes
    const incomingEdges = allEdges.filter(edge => edge.target === node.id);
    incomingEdges.forEach(edge => {
      const sourceNode = allNodes.find(n => n.id === edge.source);
      if (sourceNode && sourceNode.data.output_data) {
        const outputKey = Object.keys(sourceNode.data.output_data)[0]; // Assuming single output
        inputData[edge.targetHandle as string] = sourceNode.data.output_data[outputKey];
      }
    });

    // Filter out any inputs that are not in the block's schema
    Object.keys(inputData).forEach(key => {
      if (!blockSchema.properties[key]) {
        delete inputData[key];
      }
    });

    console.log(`Final prepared input for ${node.data.blockType} (${node.id}):`, inputData);
    return inputData;
  };


  const runAgent = async () => {
    try {
      console.log("All nodes before formatting:", nodes);

      const formattedNodes = nodes.map(node => {
        console.log("Formatting node:", node.id, node.data.blockType);
        const inputDefault = prepareNodeInputData(node, nodes, edges);
        const inputNodes = edges
          .filter(edge => edge.target === node.id)
          .map(edge => ({
            name: edge.targetHandle || '',
            node_id: edge.source,
          }));

        const outputNodes = edges
          .filter(edge => edge.source === node.id)
          .map(edge => ({
            name: edge.sourceHandle || '',
            node_id: edge.target,
          }));

        return {
          id: node.id,
          block_id: node.data.block_id,
          input_default: inputDefault,
          input_nodes: inputNodes,
          output_nodes: outputNodes,
          metadata: { position: node.position }
        };
      });

      const links = edges.map(edge => ({
        source_id: edge.source,
        sink_id: edge.target,
        source_name: edge.sourceHandle || '',
        sink_name: edge.targetHandle || ''
      }));

      const payload = {
        id: agentId || '',
        name: 'Agent Name',
        description: 'Agent Description',
        nodes: formattedNodes,
        links: links  // Ensure this field is included
      };

      const createData = await api.createFlow(payload);
      const newAgentId = createData.id;
      setAgentId(newAgentId);
      console.log('Response from the API:', JSON.stringify(createData, null, 2));

      const executeData = await api.executeFlow(newAgentId);
      const runId = executeData.id;

      const pollExecution = async () => {
        const data = await api.getFlowExecutionInfo(newAgentId, runId);
        updateNodesWithExecutionData(data);

        if (data.every((node) => node.status === 'COMPLETED')) {
          console.log('All nodes completed execution');
        } else {
          setTimeout(pollExecution, 1000);
        }
      };

      pollExecution();

    } catch (error) {
      console.error('Error running agent:', error);
    }
  };


const updateNodesWithExecutionData = (executionData: any[]) => {
  setNodes((nds) =>
    nds.map((node) => {
      const nodeExecution = executionData.find((exec) => exec.node_id === node.id);
      if (nodeExecution) {
        return {
          ...node,
          data: {
            ...node.data,
            status: nodeExecution.status,
            output_data: nodeExecution.output_data,
            isPropertiesOpen: true,
          },
        };
      }
      return node;
    })
  );
};

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  return (
    <div style={{ height: '100vh', width: '100%' }}>
      <button
        onClick={toggleSidebar}
        style={{
          position: 'absolute',
          left: isSidebarOpen ? '260px' : '10px',
          top: '10px',
          zIndex: 5,
          transition: 'left 0.3s'
        }}
      >
        {isSidebarOpen ? 'Hide Sidebar' : 'Show Sidebar'}
      </button>
      <Sidebar isOpen={isSidebarOpen} availableNodes={availableNodes} addNode={addNode} />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
      >
        <div style={{ position: 'absolute', right: 10, top: 10, zIndex: 4 }}>
          <button onClick={runAgent}>Run Agent</button>
        </div>
      </ReactFlow>
    </div>
  );
};

export default FlowEditor;
