import { XYPosition } from "reactflow";
import { ObjectSchema } from "./types";

export default class AutoGPTServerAPI {
  private baseUrl: string;
  private wsUrl: string;
  private socket: WebSocket | null = null;
  private messageHandlers: { [key: string]: (data: any) => void } = {};

  constructor(
    baseUrl: string = process.env.AGPT_SERVER_URL || "http://localhost:8000/api"
  ) {
    this.baseUrl = baseUrl;
    this.wsUrl = `ws://${new URL(this.baseUrl).host}/ws`;
  }

  async getBlocks(): Promise<Block[]> {
    return await this._get("/blocks");
  }

  async listGraphIDs(): Promise<string[]> {
    return this._get("/graphs")
  }

  async listTemplates(): Promise<GraphMeta[]> {
    return this._get("/templates")
  }

  async getGraph(id: string, version?: number): Promise<Graph> {
    const query = version !== undefined ? `?version=${version}` : "";
    return this._get(`/graphs/${id}` + query);
  }

  async getTemplate(id: string, version?: number): Promise<Graph> {
    const query = version !== undefined ? `?version=${version}` : "";
    return this._get(`/templates/${id}` + query);
  }

  async getGraphAllVersions(id: string): Promise<Graph[]> {
    return this._get(`/graphs/${id}/versions`);
  }

  async getTemplateAllVersions(id: string): Promise<Graph[]> {
    return this._get(`/templates/${id}/versions`);
  }

  async createGraph(graphCreateBody: GraphCreatable): Promise<Graph>;
  async createGraph(fromTemplateID: string, templateVersion: number): Promise<Graph>;
  async createGraph(
    graphOrTemplateID: GraphCreatable | string, templateVersion?: number
  ): Promise<Graph> {
    let requestBody: GraphCreateRequestBody;

    if (typeof(graphOrTemplateID) == "string") {
      if (templateVersion == undefined) {
        throw new Error("templateVersion not specified")
      }
      requestBody = {
        template_id: graphOrTemplateID,
        template_version: templateVersion,
      }
    } else {
      requestBody = { graph: graphOrTemplateID }
    }

    return this._request("POST", "/graphs", requestBody);
  }

  async createTemplate(templateCreateBody: GraphCreatable): Promise<Graph> {
    const requestBody: GraphCreateRequestBody = { graph: templateCreateBody };
    return this._request("POST", "/templates", requestBody);
  }

  async updateGraph(id: string, graph: GraphUpdateable): Promise<Graph> {
    return await this._request("PUT", `/graphs/${id}`, graph);
  }

  async updateTemplate(id: string, template: GraphUpdateable): Promise<Graph> {
    return await this._request("PUT", `/templates/${id}`, template);
  }

  async setGraphActiveVersion(id: string, version: number): Promise<Graph> {
    return this._request(
      "PUT", `/graphs/${id}/versions/active`, { active_graph_version: version }
    );
  }

  async executeGraph(
    id: string, inputData: { [key: string]: any } = {}
  ): Promise<GraphExecuteResponse> {
    return this._request("POST", `/graphs/${id}/execute`, inputData);
  }

  async listGraphRunIDs(graphID: string, graphVersion?: number): Promise<string[]> {
    const query = graphVersion !== undefined ? `?graph_version=${graphVersion}` : "";
    return this._get(`/graphs/${graphID}/executions` + query);
  }

  async getGraphExecutionInfo(graphID: string, runID: string): Promise<NodeExecutionResult[]> {
    return (await this._get(`/graphs/${graphID}/executions/${runID}`))
    .map((result: any) => ({
      ...result,
      add_time: new Date(result.add_time),
      queue_time: result.queue_time ? new Date(result.queue_time) : undefined,
      start_time: result.start_time ? new Date(result.start_time) : undefined,
      end_time: result.end_time ? new Date(result.end_time) : undefined,
    }));
  }

  private async _get(path: string) {
    return this._request("GET", path);
  }

  private async _request(
    method: "GET" | "POST" | "PUT" | "PATCH",
    path: string,
    payload?: { [key: string]: any },
  ) {
    if (method != "GET") {
      console.debug(`${method} ${path} payload:`, payload);
    }

    const response = await fetch(
      this.baseUrl + path,
      method != "GET" ? {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      } : undefined
    );
    const response_data = await response.json();

    if (!response.ok) {
      console.warn(
        `${method} ${path} returned non-OK response:`, response_data.detail, response
      );
      throw new Error(`HTTP error ${response.status}! ${response_data.detail}`);
    }
    return response_data;
  }

  connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.socket = new WebSocket(this.wsUrl);

      this.socket.onopen = () => {
        console.log('WebSocket connection established');
        resolve();
      };

      this.socket.onclose = (event) => {
        console.log('WebSocket connection closed', event);
        this.socket = null;
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };

      this.socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (this.messageHandlers[message.method]) {
          this.messageHandlers[message.method](message.data);
        }
      };
    });
  }

  disconnectWebSocket() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.close();
    }
  }

  sendWebSocketMessage(method: string, data: any) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ method, data }));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  onWebSocketMessage(method: string, handler: (data: any) => void) {
    this.messageHandlers[method] = handler;
  }

  subscribeToExecution(graphId: string) {
    this.sendWebSocketMessage('subscribe', { graph_id: graphId });
  }

  runGraph(graphId: string, data: any = {}) {
    this.sendWebSocketMessage('run_graph', { graph_id: graphId, data });
  }
}

/* Mirror of autogpt_server/data/block.py:Block */
export type Block = {
  id: string;
  name: string;
  description: string;
  inputSchema: ObjectSchema;
  outputSchema: ObjectSchema;
};

/* Mirror of autogpt_server/data/graph.py:Node */
export type Node = {
  id: string;
  block_id: string;
  input_default: { [key: string]: any };
  input_nodes: Array<{ name: string, node_id: string }>;
  output_nodes: Array<{ name: string, node_id: string }>;
  metadata: {
    position: XYPosition;
    [key: string]: any;
  };
};

/* Mirror of autogpt_server/data/graph.py:Link */
export type Link = {
  id: string;
  source_id: string;
  sink_id: string;
  source_name: string;
  sink_name: string;
}

export type LinkCreatable = Omit<Link, "id"> & {
  id?: string;
}

/* Mirror of autogpt_server/data/graph.py:GraphMeta */
export type GraphMeta = {
  id: string;
  version: number;
  is_active: boolean;
  is_template: boolean;
  name: string;
  description: string;
}

/* Mirror of autogpt_server/data/graph.py:Graph */
export type Graph = GraphMeta & {
  nodes: Array<Node>;
  links: Array<Link>;
};

export type GraphUpdateable = Omit<
  Graph,
  "version" | "is_active" | "is_template" | "links"
> & {
  version?: number;
  is_active?: boolean;
  is_template?: boolean;
  links: Array<LinkCreatable>;
}

export type GraphCreatable = Omit<GraphUpdateable, "id"> & { id?: string }

export type GraphCreateRequestBody = {
  template_id: string;
  template_version: number;
} | {
  graph: GraphCreatable;
}

/* Derived from autogpt_server/executor/manager.py:ExecutionManager.add_execution */
export type GraphExecuteResponse = {
  /* ID of the initiated run */
  id: string;
  /* List of node executions */
  executions: Array<{ id: string, node_id: string }>;
};

/* Mirror of autogpt_server/data/execution.py:ExecutionResult */
export type NodeExecutionResult = {
  graph_exec_id: string;
  node_exec_id: string;
  graph_id: string;
  graph_version: number;
  node_id: string;
  status: 'INCOMPLETE' | 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  input_data: { [key: string]: any };
  output_data: { [key: string]: Array<any> };
  add_time: Date;
  queue_time?: Date;
  start_time?: Date;
  end_time?: Date;
};
