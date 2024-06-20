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
} from 'reactflow';
import 'reactflow/dist/style.css';
import Modal from 'react-modal';
import CustomNode from './CustomNode';
import './index.css';

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];
const nodeTypes: NodeTypes = {
  custom: CustomNode,
};

interface AvailableNode {
  id: string;
  name: string;
  description: string;
  inputSchema?: { properties: { [key: string]: any } };
  outputSchema?: { properties: { [key: string]: any } };
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

  useEffect(() => {
    fetch('http://192.168.0.215:8000/blocks')
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        setAvailableNodes(data);
        setLoadingStatus('loaded');
      })
      .catch(error => {
        console.error('Error fetching nodes:', error);
        setLoadingStatus('failed');
      });
  }, []);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
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
        hardcodedValues: {}, // Added hardcodedValues to store hardcoded inputs
        setHardcodedValues: (values: { [key: string]: any }) => {
          setNodes((nds) => nds.map((node) =>
            node.id === nodeId.toString()
              ? { ...node, data: { ...node.data, hardcodedValues: values } }
              : node
          ));
        },
        block_id: nodeSchema?.id || '',
      },
      position: { x: Math.random() * 400, y: Math.random() * 400 },
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
            ? { ...node, data: { ...node.data, title, description, label: title, variableName, variableValue, printVariable } }
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

  const runAgent = async () => {
    try {
      // Format each node as required by the backend
      const formattedNodes = nodes.map(node => ({
        id: node.id,
        block_id: node.data.block_id,
        input_default: node.data.hardcodedValues || {},
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
      }));
  
      // Create agent payload
      const payload = {
        id: '',
        name: 'Agent Name',
        description: 'Agent Description',
        nodes: formattedNodes,
      };
  
      console.log('Agent creation payload:', payload);
  
      // Create the agent
      const createResponse = await fetch('http://192.168.0.215:8000/agents', {
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
  
      // Collect the initial inputs for the agent's nodes
      const nodeInput = nodes.reduce((acc, node) => {
        if (node.data.hardcodedValues && Object.keys(node.data.hardcodedValues).length > 0) {
          acc[node.id] = node.data.hardcodedValues;
        }
        return acc;
      }, {} as { [key: string]: any });
  
      // Adjust node input for PrintingBlock to include 'text' directly
      const nodeInputForExecution = Object.keys(nodeInput).reduce((acc, key) => {
        acc[key] = nodeInput[key];
        return acc;
      }, {} as { [key: string]: any });
  
      // Ensure nodeInput for PrintingBlock includes 'text'
      if (!nodeInputForExecution['1'] || !nodeInputForExecution['1'].text) {
        nodeInputForExecution['1'] = { text: 'hello' };
      }
  
      console.log('Node input for execution:', nodeInputForExecution);
  
      // Payload for executing the agent
      const executeResponse = await fetch(`http://192.168.0.215:8000/agents/${agentId}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(nodeInputForExecution['1']),
      });
  
      if (!executeResponse.ok) {
        throw new Error(`HTTP error! Status: ${executeResponse.status}`);
      }
  
      const executeData = await executeResponse.json();
      const runId = executeData.run_id;
  
      console.log(`Agent ${agentId} is executing with run ID ${runId}`);
  
      const watchExecution = async () => {
        const watchResponse = await fetch(`http://192.168.0.215:8000/agents/${agentId}/executions/${runId}`);
  
        if (!watchResponse.ok) {
          throw new Error(`HTTP error! Status: ${watchResponse.status}`);
        }
  
        const executionData = await watchResponse.json();
        console.log('Execution data:', executionData);
      };
  
      watchExecution();
    } catch (error) {
      console.error('Error running agent:', error);
    }
  };

  return (
    <div style={{ height: '100vh', position: 'relative', backgroundColor: '#121212' }}>
      <div style={{ position: 'absolute', top: '20px', left: isSidebarOpen ? '400px' : '20px', zIndex: 10, transition: 'left 0.3s ease' }}>
        <button
          onClick={toggleSidebar}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#007bff',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Nodes
        </button>
        <button
          onClick={runAgent}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: 'green',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            marginLeft: '10px',
          }}
        >
          Run
        </button>
        {agentId && (
          <span style={{ marginLeft: '10px', color: '#fff', fontSize: '16px' }}>
            Agent ID: {agentId}
          </span>
        )}
      </div>
      <div style={{ height: '100%', width: '100%' }}>
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
