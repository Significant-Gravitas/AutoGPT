"use client";

import React, { useState, useCallback, useEffect } from 'react';
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
  EdgeRemoveChange,
} from 'reactflow';
import 'reactflow/dist/style.css';
import Modal from 'react-modal';
import CustomNode from './CustomNode';
import './flow.css';

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];
const nodeTypes: NodeTypes = {
  custom: CustomNode,
};

interface AvailableNode {
  id: string;
  name: string;
  description: string;
  inputSchema?: { properties: { [key: string]: any }; required?: string[] };
  outputSchema?: { properties: { [key: string]: any } };
}

interface ExecData {
  node_id: string;
  status: string;
  output_data: any;
}

const Flow: React.FC = () => {
  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>(initialEdges);
  const [nodeId, setNodeId] = useState<number>(1);
  const [modalIsOpen, setModalIsOpen] = useState<boolean>(false);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [title, setTitle] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [variableName, setVariableName] = useState<string>('');
  const [variableValue, setVariableValue] = useState<string>('');
  const [printVariable, setPrintVariable] = useState<string>('');
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [availableNodes, setAvailableNodes] = useState<AvailableNode[]>([]);
  const [loadingStatus, setLoadingStatus] = useState<'loading' | 'failed' | 'loaded'>('loading');
  const [agentId, setAgentId] = useState<string | null>(null);

  const apiUrl = 'http://localhost:8000'

  useEffect(() => {
    fetch(`${apiUrl}/blocks`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        setAvailableNodes(data.map((node: AvailableNode) => ({
          ...node,
          description: typeof node.description === 'object' ? JSON.stringify(node.description) : node.description,
        })));
        setLoadingStatus('loaded');
      })
      .catch(error => {
        console.error('Error fetching nodes:', error);
        setLoadingStatus('failed');
      });
  }, []);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds).map(node => ({
      ...node,
      data: {
        ...node.data,
        metadata: {
          ...node.data.metadata,
          position: node.position
        }
      }
    }))),
    []
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      const removedEdges = changes.filter((change): change is EdgeRemoveChange => change.type === 'remove');
      setEdges((eds) => applyEdgeChanges(changes, eds));

      if (removedEdges.length > 0) {
        setNodes((nds) =>
          nds.map((node) => {
            const updatedConnections = node.data.connections.filter(
              (conn: string) =>
                !removedEdges.some((edge) => edge.id && conn.includes(edge.id))
            );
            return { ...node, data: { ...node.data, connections: updatedConnections } };
          })
        );
      }
    },
    []
  );

  const onConnect: OnConnect = useCallback(
    (connection) => {
      setEdges((eds) => addEdge(connection, eds));
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === connection.source) {
            const connections = node.data.connections || [];
            connections.push(`${node.data.title} ${connection.sourceHandle} -> ${connection.targetHandle}`);
            return { ...node, data: { ...node.data, connections } };
          }
          if (node.id === connection.target) {
            const connections = node.data.connections || [];
            connections.push(`${connection.sourceHandle} -> ${node.data.title} ${connection.targetHandle}`);
            return { ...node, data: { ...node.data, connections } };
          }
          return node;
        })
      );
    },
    [setEdges, setNodes]
  );

  const addNode = (type: string, label: string, description: string) => {
    const nodeSchema = availableNodes.find(node => node.name === label);
    const position = { x: Math.random() * 400, y: Math.random() * 400 };

    const newNode: Node = {
      id: nodeId.toString(),
      type: 'custom',
      data: {
        label: label,
        title: `${type} ${nodeId}`,
        description: `${description}`,
        inputSchema: nodeSchema?.inputSchema,
        outputSchema: nodeSchema?.outputSchema,
        connections: [],
        variableName: '',
        variableValue: '',
        printVariable: '',
        setVariableName,
        setVariableValue,
        setPrintVariable,
        hardcodedValues: {},
        setHardcodedValues: (values: { [key: string]: any }) => {
          setNodes((nds) => nds.map((node) =>
            node.id === nodeId.toString()
              ? { ...node, data: { ...node.data, hardcodedValues: values } }
              : node
          ));
        },
        block_id: nodeSchema?.id || '',
        metadata: {
          position // Store position in metadata
        }
      },
      position,
    };
    setNodes((nds) => [...nds, newNode]);
    setNodeId((id) => id + 1);
  };

  const closeModal = () => {
    setModalIsOpen(false);
    setSelectedNode(null);
  };

  const saveNodeData = () => {
    if (selectedNode) {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === selectedNode.id
            ? {
                ...node,
                data: {
                  ...node.data,
                  title,
                  description,
                  label: title,
                  variableName,
                  variableValue: typeof variableValue === 'object' ? JSON.stringify(variableValue) : variableValue,
                  printVariable: typeof printVariable === 'object' ? JSON.stringify(printVariable) : printVariable,
                },
              }
            : node
        )
      );
      closeModal();
    }
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const filteredNodes = availableNodes.filter(node => node.name.toLowerCase().includes(searchQuery.toLowerCase()));

  const prepareNodeInputData = (node: Node, allNodes: Node[], allEdges: Edge[]) => {
    const nodeSchema = availableNodes.find(n => n.id === node.data.block_id);
    if (!nodeSchema || !nodeSchema.inputSchema) return {};

    let inputData: { [key: string]: any } = {};
    const inputProperties = nodeSchema.inputSchema.properties;
    const requiredProperties = nodeSchema.inputSchema.required || [];

    // Initialize inputData with default values for all required properties
    requiredProperties.forEach(prop => {
      inputData[prop] = node.data.hardcodedValues[prop] || '';
    });

    Object.keys(inputProperties).forEach(prop => {
      const inputEdge = allEdges.find(edge => edge.target === node.id && edge.targetHandle === prop);
      if (inputEdge) {
        const sourceNode = allNodes.find(n => n.id === inputEdge.source);
        inputData[prop] = sourceNode?.data.output_data || sourceNode?.data.hardcodedValues[prop] || '';
      } else if (node.data.hardcodedValues && node.data.hardcodedValues[prop]) {
        inputData[prop] = node.data.hardcodedValues[prop];
      }
    });

    return inputData;
  };

  const updateNodeData = (execData: ExecData) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === execData.node_id) {
          return {
            ...node,
            data: {
              ...node.data,
              status: execData.status,
              output_data: execData.output_data,
              isPropertiesOpen: true, // Open the properties
            },
          };
        }
        return node;
      })
    );
  };

  const runAgent = async () => {
    try {
      const formattedNodes = nodes.map(node => ({
        id: node.id,
        block_id: node.data.block_id,
        input_default: prepareNodeInputData(node, nodes, edges),
        input_nodes: edges.filter(edge => edge.target === node.id).reduce((acc, edge) => {
          if (edge.targetHandle) {
            acc[edge.targetHandle] = edge.source;
          }
          return acc;
        }, {} as { [key: string]: string }),
        output_nodes: edges.filter(edge => edge.source === node.id).reduce((acc, edge) => {
          if (edge.sourceHandle) {
            acc[edge.sourceHandle] = edge.target;
          }
          return acc;
        }, {} as { [key: string]: string }),
        metadata: node.data.metadata,
        connections: node.data.connections // Ensure connections are preserved
      }));

      const payload = {
        id: '',
        name: 'Agent Name',
        description: 'Agent Description',
        nodes: formattedNodes,
      };

      const createResponse = await fetch(`${apiUrl}/agents`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!createResponse.ok) {
        throw new Error(`HTTP error! Status: ${createResponse.status}`);
      }

      const createData = await createResponse.json();
      const agentId = createData.id;
      setAgentId(agentId);

      const responseNodes = createData.nodes.map((node: any) => {
        const block = availableNodes.find(n => n.id === node.block_id);
        const connections = edges.filter(edge => edge.source === node.id || edge.target === node.id).map(edge => ({
          id: edge.id,
          source: edge.source,
          sourceHandle: edge.sourceHandle,
          target: edge.target,
          targetHandle: edge.targetHandle
        }));
        return {
          id: node.id,
          type: 'custom',
          position: node.metadata.position,
          data: {
            label: block?.name || 'Unknown',
            title: `${block?.name || 'Unknown'}`,
            description: `${block?.description || ''}`,
            inputSchema: block?.inputSchema,
            outputSchema: block?.outputSchema,
            connections: connections.map(c => `${c.source}-${c.sourceHandle} -> ${c.target}-${c.targetHandle}`),
            variableName: '',
            variableValue: '',
            printVariable: '',
            setVariableName,
            setVariableValue,
            setPrintVariable,
            hardcodedValues: node.input_default,
            setHardcodedValues: (values: { [key: string]: any }) => {
              setNodes((nds) => nds.map((n) =>
                n.id === node.id
                  ? { ...n, data: { ...n.data, hardcodedValues: values } }
                  : n
              ));
            },
            block_id: node.block_id,
            metadata: node.metadata
          },
        };
      });

      const newEdges = createData.nodes.flatMap((node: any) => {
        return Object.entries(node.output_nodes).map(([sourceHandle, targetNodeId]) => ({
          id: `${node.id}-${sourceHandle}-${targetNodeId}`,
          source: node.id,
          sourceHandle: sourceHandle,
          target: targetNodeId,
          targetHandle: Object.keys(node.input_nodes).find(key => node.input_nodes[key] === targetNodeId) || '',
        }));
      });

      setNodes(responseNodes);
      setEdges(newEdges);

      const initialNodeInput = nodes.reduce((acc, node) => {
        acc[node.id] = prepareNodeInputData(node, nodes, edges);
        return acc;
      }, {} as { [key: string]: any });

      const nodeInputForExecution = Object.keys(initialNodeInput).reduce((acc, key) => {
        const blockId = nodes.find(node => node.id === key)?.data.block_id;
        const nodeSchema = availableNodes.find(n => n.id === blockId);
        if (nodeSchema && nodeSchema.inputSchema) {
          Object.keys(nodeSchema.inputSchema.properties).forEach(prop => {
            acc[prop] = initialNodeInput[key][prop];
          });
        }
        return acc;
      }, {} as { [key: string]: any });

      const executeResponse = await fetch(`${apiUrl}/agents/${agentId}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(nodeInputForExecution),
      });

      if (!executeResponse.ok) {
        throw new Error(`HTTP error! Status: ${executeResponse.status}`);
      }

      const executeData = await executeResponse.json();
      const runId = executeData.run_id;

      const startPolling = () => {
        const endTime = Date.now() + 60000;

        const poll = async () => {
          if (Date.now() >= endTime) {
            console.log('Polling timeout reached.');
            return;
          }

          try {
            const response = await fetch(`${apiUrl}/agents/${agentId}/executions/${runId}`);
            if (!response.ok) {
              throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            data.forEach(updateNodeData);

            const allCompleted = data.every((exec: any) => exec.status === 'COMPLETED');
            if (allCompleted) {
              console.log('All nodes are completed.');
              return;
            }

            setTimeout(poll, 100);
          } catch (error) {
            console.error('Error during polling:', error);
            setTimeout(poll, 100);
          }
        };

        poll();
      };

      startPolling();
    } catch (error) {
      console.error('Error running agent:', error);
    }
  };

  return (
    <div className="flow-container">
      <div className={`flow-controls ${isSidebarOpen ? 'open' : ''}`}>
        <button className="nodes-button" onClick={toggleSidebar}>
          Nodes
        </button>
        <button className="run-button" onClick={runAgent}>
          Run
        </button>
        {agentId && (
          <span style={{ marginLeft: '10px', color: '#fff', fontSize: '16px' }}>
            Agent ID: {agentId}
          </span>
        )}
      </div>
      <div className="flow-wrapper">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
        />
      </div>
      {selectedNode && (
        <Modal isOpen={modalIsOpen} onRequestClose={closeModal} contentLabel="Node Info" className="modal" overlayClassName="overlay">
          <h2>Edit Node</h2>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              saveNodeData();
            }}
          >
            <div>
              <label>
                Title:
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  required
                />
              </label>
            </div>
            <div>
              <label>
                Description:
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  required
                />
              </label>
            </div>
            {selectedNode.data.title.includes('Variable') && (
              <>
                <div>
                  <label>
                    Variable Name:
                    <input
                      type="text"
                      value={variableName}
                      onChange={(e) => setVariableName(e.target.value)}
                      required
                    />
                  </label>
                </div>
                <div>
                  <label>
                    Variable Value:
                    <input
                      type="text"
                      value={variableValue}
                      onChange={(e) => setVariableValue(e.target.value)}
                      required
                    />
                  </label>
                </div>
              </>
            )}
            {selectedNode.data.title.includes('Print') && (
              <>
                <div>
                  <label>
                    Variable to Print:
                    <input
                      type="text"
                      value={printVariable}
                      onChange={(e) => setPrintVariable(e.target.value)}
                      required
                    />
                  </label>
                </div>
              </>
            )}
            <button type="submit">Save</button>
            <button type="button" onClick={closeModal}>
              Cancel
            </button>
          </form>
        </Modal>
      )}
      <div className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <h3 style={{ margin: '0 0 10px 0' }}>Nodes</h3>
        <input
          type="text"
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            padding: '10px',
            fontSize: '16px',
            backgroundColor: '#333',
            color: '#e0e0e0',
            border: '1px solid #555',
            borderRadius: '4px',
            marginBottom: '10px',
            width: 'calc(100% - 22px)',
            boxSizing: 'border-box',
          }}
        />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {loadingStatus === 'loading' && <p>Loading...</p>}
          {loadingStatus === 'failed' && <p>Failed To Load Nodes</p>}
          {loadingStatus === 'loaded' && filteredNodes.map(node => (
            <div key={node.id} style={sidebarNodeRowStyle}>
              <div>
                <strong>{node.name}</strong>
                <p>{node.description}</p>
              </div>
              <button
                onClick={() => addNode(node.name, node.name, node.description)}
                style={sidebarButtonStyle}
              >
                Add
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const sidebarNodeRowStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  backgroundColor: '#444',
  padding: '10px',
  borderRadius: '4px',
};

const sidebarButtonStyle = {
  padding: '10px 20px',
  fontSize: '16px',
  backgroundColor: '#007bff',
  color: '#fff',
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
};

export default Flow;
