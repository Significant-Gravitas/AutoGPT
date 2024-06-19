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

const initialNodes: Node[] = [
  { id: '1', type: 'custom', data: { label: 'Start Node', title: 'Start Node', description: 'Start Node' }, position: { x: 250, y: 5 } },
];

const initialEdges: Edge[] = [];

const nodeTypes: NodeTypes = {
  custom: CustomNode,
};

interface AvailableNode {
  id: string;
  name: string;
  inputSchema?: { properties: { [key: string]: any } };
  outputSchema?: { properties: { [key: string]: any } };
}

const Flow: React.FC = () => {
  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>(initialEdges);
  const [nodeId, setNodeId] = useState<number>(2);
  const [modalIsOpen, setModalIsOpen] = useState<boolean>(false);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [title, setTitle] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [variableName, setVariableName] = useState<string>('');
  const [variableValue, setVariableValue] = useState<string>('');
  const [printVariable, setPrintVariable] = useState<string>('');
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [isPropertiesSidebarOpen, setIsPropertiesSidebarOpen] = useState<boolean>(false);
  const [propertiesNode, setPropertiesNode] = useState<Node | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [availableNodes, setAvailableNodes] = useState<AvailableNode[]>([]);

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
      })
      .catch(error => console.error('Error fetching nodes:', error));
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
    (connection) => setEdges((eds) => addEdge(connection, eds)),
    []
  );

  const addNode = (type: string, label: string) => {
    const nodeSchema = availableNodes.find(node => node.name === label);

    const newNode: Node = {
      id: nodeId.toString(),
      type: 'custom',
      data: {
        label: label,
        title: `${type} ${nodeId}`,
        description: `${type} Description ${nodeId}`,
        inputSchema: nodeSchema?.inputSchema,
        outputSchema: nodeSchema?.outputSchema,
        variableName: '',
        variableValue: '',
        printVariable: '',
        setVariableName,
        setVariableValue,
        setPrintVariable,
        openPropertiesSidebar: (node: Node) => openPropertiesSidebar(node),
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

  const openPropertiesSidebar = (node: Node) => {
    console.log(node);
    setPropertiesNode(node);
    setIsPropertiesSidebarOpen(true);
  };
    
  const closePropertiesSidebar = () => {
    setIsPropertiesSidebarOpen(false);
    setPropertiesNode(null);
  };
    
  const getNodeConnections = (nodeId: string) => {
    const inputs = edges.filter(edge => edge.target === nodeId).length;
    const outputs = edges.filter(edge => edge.source === nodeId).length;
    return { inputs, outputs };
  };

  const filteredNodes = availableNodes.filter(node => node.name.toLowerCase().includes(searchQuery.toLowerCase()));

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
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
          {filteredNodes.map(node => (
            <button
              key={node.id}
              onClick={() => addNode(node.name, node.name)}
              style={sidebarButtonStyle}
            >
              {`Add ${node.name}`}
            </button>
          ))}
        </div>
      </div>
      {propertiesNode && (
        <div className={`properties-sidebar ${isPropertiesSidebarOpen ? 'open' : ''}`}>
          <h3>Node Properties</h3>
          <p><strong>Name:</strong> {propertiesNode?.data?.label}</p>
          <p><strong>Title:</strong> {propertiesNode?.data?.title}</p>
          <p><strong>Description:</strong> {propertiesNode?.data?.description}</p>
          <p><strong>Inputs:</strong> {getNodeConnections(propertiesNode.id).inputs}</p>
          <p><strong>Outputs:</strong> {getNodeConnections(propertiesNode.id).outputs}</p>
          {propertiesNode.data?.variableName !== undefined && (
            <p><strong>Variable Name:</strong> {propertiesNode.data.variableName}</p>
          )}
          {propertiesNode.data?.variableValue !== undefined && (
            <p><strong>Variable Value:</strong> {propertiesNode.data.variableValue}</p>
          )}
          {propertiesNode.data?.printVariable !== undefined && (
            <p><strong>Variable to Print:</strong> {propertiesNode.data.printVariable}</p>
          )}
          <button onClick={closePropertiesSidebar}>Close</button>
        </div>
      )}
    </div>
  );
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
