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
import AutoGPTServerAPI, { Block, Graph } from '@/lib/autogpt_server_api';
import { ObjectSchema } from '@/lib/types';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ChevronRight, ChevronLeft } from "lucide-react";
import { deepEquals } from '@/lib/utils';
import { beautifyString } from '@/lib/utils';


type CustomNodeData = {
  blockType: string;
  title: string;
  inputSchema: ObjectSchema;
  outputSchema: ObjectSchema;
  hardcodedValues: { [key: string]: any };
  setHardcodedValues: (values: { [key: string]: any }) => void;
  connections: Array<{ source: string; sourceHandle: string; target: string; targetHandle: string }>;
  isOutputOpen: boolean;
  status?: string;
  output_data?: any;
  block_id: string;
  backend_id?: string;
};

const Sidebar: React.FC<{ isOpen: boolean, availableNodes: Block[], addNode: (id: string, name: string) => void }> =
  ({ isOpen, availableNodes, addNode }) => {
    const [searchQuery, setSearchQuery] = useState('');

    if (!isOpen) return null;

    const filteredNodes = availableNodes.filter(node =>
      node.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
      <div className={`sidebar dark-theme ${isOpen ? 'open' : ''}`}>
        <h3>Nodes</h3>
        <Input
          type="text"
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {filteredNodes.map((node) => (
          <div key={node.id} className="sidebarNodeRowStyle dark-theme">
            <span>{beautifyString(node.name).replace(/Block$/, '')}</span>
            <Button onClick={() => addNode(node.id, node.name)}>Add</Button>
          </div>
        ))}
      </div>
    );
  };

const FlowEditor: React.FC<{
  flowID?: string;
  template?: boolean;
  className?: string;
}> = ({ flowID, template, className }) => {
  const [nodes, setNodes] = useState<Node<CustomNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [nodeId, setNodeId] = useState<number>(1);
  const [availableNodes, setAvailableNodes] = useState<Block[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [savedAgent, setSavedAgent] = useState<Graph | null>(null);
  const [agentDescription, setAgentDescription] = useState<string>('');
  const [agentName, setAgentName] = useState<string>('');

  const apiUrl = process.env.AGPT_SERVER_URL!;
  const api = useMemo(() => new AutoGPTServerAPI(apiUrl), [apiUrl]);

  useEffect(() => {
    api.connectWebSocket()
      .then(() => {
        console.log('WebSocket connected');
        api.onWebSocketMessage('execution_event', (data) => {
          updateNodesWithExecutionData([data]);
        });
      })
      .catch((error) => {
        console.error('Failed to connect WebSocket:', error);
      });

    return () => {
      api.disconnectWebSocket();
    };
  }, [api]);

  useEffect(() => {
    api.getBlocks()
      .then(blocks => setAvailableNodes(blocks))
      .catch();
  }, []);

  // Load existing graph
  useEffect(() => {
    if (!flowID || availableNodes.length == 0) return;

    (template ? api.getTemplate(flowID) : api.getGraph(flowID))
      .then(graph => loadGraph(graph));
  }, [flowID, template, availableNodes]);

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

  const onEdgesDelete = useCallback(
  (edgesToDelete: Edge[]) => {
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
                  edge.sourceHandle === conn.sourceHandle &&
                  edge.targetHandle === conn.targetHandle
              )
          ),
        },
      }))
    );
  },
  [setNodes]
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
        isOutputOpen: false,
        block_id: blockId,
      },
    };

    setNodes((nds) => [...nds, newNode]);
    setNodeId((prevId) => prevId + 1);
  };

  function loadGraph(graph: Graph) {
    setSavedAgent(graph);
    setAgentName(graph.name);
    setAgentDescription(graph.description);

    setNodes(graph.nodes.map(node => {
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
          hardcodedValues: node.input_default,
          setHardcodedValues: (values: { [key: string]: any; }) => {
            setNodes((nds) => nds.map((node) => node.id === newNode.id
              ? { ...node, data: { ...node.data, hardcodedValues: values } }
              : node
            ));
          },
          connections: [],
          isOutputOpen: false,
        },
      };
      return newNode;
    }));

    setEdges(graph.links.map(link => ({
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

    console.log(`Final prepared input for ${node.data.blockType} (${node.id}):`, inputData);
    return inputData;
  };

  async function saveAgent (asTemplate: boolean = false) {
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
          status: undefined,
        },
      }))
    );
    await new Promise((resolve) => setTimeout(resolve, 100));
    console.log("All nodes before formatting:", nodes);
    const blockIdToNodeIdMap = {};

    const formattedNodes = nodes.map(node => {
      nodes.forEach(node => {
        const key = `${node.data.block_id}_${node.position.x}_${node.position.y}`;
        blockIdToNodeIdMap[key] = node.id;
      });
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
      id: savedAgent?.id!,
      name: agentName || 'Agent Name',
      description: agentDescription || 'Agent Description',
      nodes: formattedNodes,
      links: links  // Ensure this field is included
    };

    if (savedAgent && deepEquals(payload, savedAgent)) {
      console.debug("No need to save: Graph is the same as version on server");
      return;
    } else {
      console.debug("Saving new Graph version; old vs new:", savedAgent, payload);
    }

    const newSavedAgent = savedAgent
      ? await (savedAgent.is_template
        ? api.updateTemplate(savedAgent.id, payload) 
        : api.updateGraph(savedAgent.id, payload))
      : await (asTemplate
        ? api.createTemplate(payload)
        : api.createGraph(payload));
    console.debug('Response from the API:', newSavedAgent);
    setSavedAgent(newSavedAgent);

    // Update the node IDs in the frontend
    const updatedNodes = newSavedAgent.nodes.map(backendNode => {
      const key = `${backendNode.block_id}_${backendNode.metadata.position.x}_${backendNode.metadata.position.y}`;
      const frontendNodeId = blockIdToNodeIdMap[key];
      const frontendNode = nodes.find(node => node.id === frontendNodeId);

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
    }).filter(node => node !== null);

    setNodes(updatedNodes);

    return newSavedAgent.id;
  };

  const runAgent = async () => {
    try {
      const newAgentId = await saveAgent();
      if (!newAgentId) {
        console.error('Error saving agent; aborting run');
        return;
      }

      api.subscribeToExecution(newAgentId);
      api.runGraph(newAgentId);

    } catch (error) {
      console.error('Error running agent:', error);
    }
  };



  const updateNodesWithExecutionData = (executionData: any[]) => {
    setNodes((nds) =>
      nds.map((node) => {
        const nodeExecution = executionData.find((exec) => exec.node_id === node.data.backend_id);
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
      })
    );
  };

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  return (
    <div className={className}>
    <Button
      variant="outline"
      size="icon"
      onClick={toggleSidebar}
      style={{
        position: 'fixed',
        left: isSidebarOpen ? '350px' : '10px',
        zIndex: 10000,
        backgroundColor: 'black',
        color: 'white',
      }}
    >
      {isSidebarOpen ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
    </Button>
      <Sidebar isOpen={isSidebarOpen} availableNodes={availableNodes} addNode={addNode} />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        onEdgesDelete={onEdgesDelete}
        deleteKeyCode={["Backspace", "Delete"]}
      >
        <div style={{ position: 'absolute', right: 10, zIndex: 4 }}>
          <Input
            type="text"
            placeholder="Agent Name"
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
          />
          <Input
            type="text"
            placeholder="Agent Description"
            value={agentDescription}
            onChange={(e) => setAgentDescription(e.target.value)}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>  {/* Added gap for spacing */}
            <Button onClick={() => saveAgent(savedAgent?.is_template)}>
              Save {savedAgent?.is_template ? "Template" : "Agent"}
            </Button>
            {!savedAgent?.is_template &&
              <Button onClick={runAgent}>Save & Run Agent</Button>
            }
            {!savedAgent &&
              <Button onClick={() => saveAgent(true)}>Save as Template</Button>
            }
          </div>
        </div>
      </ReactFlow>
    </div>
  );
};

export default FlowEditor;
