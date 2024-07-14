import { XYPosition } from "reactflow";
import { ObjectSchema } from "./types";

export default class AutoGPTServerAPI {
  private baseUrl: string;

  constructor(baseUrl: string = process.env.AGPT_SERVER_URL || "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async getBlocks(): Promise<Block[]> {
    try {
      const response = await fetch(`${this.baseUrl}/blocks`);
      if (!response.ok) {
        console.warn("GET /blocks returned non-OK response:", response);
        throw new Error(`HTTP error ${response.status}!`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching blocks:', error);
      throw error;
    }
  }

  async listFlowIDs(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/graphs`);
      if (!response.ok) {
        console.warn("GET /graphs returned non-OK response:", response);
        throw new Error(`HTTP error ${response.status}!`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching flows:', error);
      throw error;
    }
  }

  async getFlow(id: string, version?: number): Promise<Flow> {
    let path = `/graphs/${id}`;
    if (version !== undefined) {
      path += `?version=${version}`;
    }
    try {
      const response = await fetch(this.baseUrl + path);
      if (!response.ok) {
        console.warn(`GET ${path} returned non-OK response:`, response);
        throw new Error(`HTTP error ${response.status}!`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching flow:', error);
      throw error;
    }
  }

  async getFlowAllVersions(id: string): Promise<Flow[]> {
    let path = `/graphs/${id}/versions`;
    try {
      const response = await fetch(this.baseUrl + path);
      if (!response.ok) {
        console.warn(`GET ${path} returned non-OK response:`, response);
        throw new Error(`HTTP error ${response.status}!`);
      }
      return await response.json();
    } catch (error) {
      console.error(`Error fetching flow ${id} versions:`, error);
      throw error;
    }
  }

  async createFlow(flowCreateBody: FlowCreatable): Promise<Flow>;
  async createFlow(fromTemplateID: string, templateVersion: number): Promise<Flow>;
  async createFlow(
    flowOrTemplateID: FlowCreatable | string, templateVersion?: number
  ): Promise<Flow> {
    let requestBody: FlowCreateRequestBody;
    if (typeof(flowOrTemplateID) == "string") {
      if (templateVersion == undefined) {
        throw new Error("templateVersion not specified")
      }
      requestBody = {
        template_id: flowOrTemplateID,
        template_version: templateVersion,
      }
    } else {
      requestBody = { graph: flowOrTemplateID }
    }
    console.debug("POST /graphs payload:", requestBody);

    try {
      const response = await fetch(`${this.baseUrl}/graphs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      const response_data = await response.json();
      if (!response.ok) {
        console.warn(
          `POST /graphs returned non-OK response:`, response_data.detail, response
        );
        throw new Error(`HTTP error ${response.status}! ${response_data.detail}`)
      }
      return response_data;
    } catch (error) {
      console.error("Error storing flow:", error);
      throw error;
    }
  }

  async updateFlow(flowID: string, flow: Flow): Promise<Flow> {
    const path = `/graphs/${flowID}`;
    console.debug(`PUT ${path} payload:`, flow);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(flow),
      });
      const response_data = await response.json();
      if (!response.ok) {
        console.warn(
          `PUT ${path} returned non-OK response:`, response_data.detail, response
        );
        throw new Error(`HTTP error ${response.status}! ${response_data.detail}`)
      }
      return response_data;
    } catch (error) {
      console.error("Error updating flow:", error);
      throw error;
    }
  }

  async setFlowActiveVersion(flowID: string, version: number): Promise<Flow> {
    const path = `/graphs/${flowID}/versions/active`;
    const payload: { active_graph_version: number } = { active_graph_version: version };
    console.debug(`PUT ${path} payload:`, payload);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      const response_data = await response.json();
      if (!response.ok) {
        console.warn(
          `PUT ${path} returned non-OK response:`, response_data.detail, response
        );
        throw new Error(`HTTP error ${response.status}! ${response_data.detail}`)
      }
      return response_data;
    } catch (error) {
      console.error("Error updating flow:", error);
      throw error;
    }
  }

  async executeFlow(
    flowId: string, inputData: { [key: string]: any } = {}
  ): Promise<FlowExecuteResponse> {
    const path = `/graphs/${flowId}/execute`;
    console.debug(`POST ${path}`);
    try {
      const response = await fetch(this.baseUrl + path, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData),
      });
      const response_data = await response.json();
      if (!response.ok) {
        console.warn(
          `POST ${path} returned non-OK response:`, response_data.detail, response
        );
        throw new Error(`HTTP error ${response.status}! ${response_data.detail}`)
      }
      return response_data;
    } catch (error) {
      console.error("Error executing flow:", error);
      throw error;
    }
  }

  async listFlowRunIDs(flowId: string, flowVersion?: number): Promise<string[]> {
    let path = `/graphs/${flowId}/executions`;
    if (flowVersion !== undefined) {
      path += `?graph_version=${flowVersion}`;
    }
    try {
      const response = await fetch(this.baseUrl + path);
      if (!response.ok) {
        console.warn(`GET ${path} returned non-OK response:`, response);
        throw new Error(`HTTP error ${response.status}!`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching flow runs:', error);
      throw error;
    }
  }

  async getFlowExecutionInfo(flowId: string, runId: string): Promise<NodeExecutionResult[]> {
    const path = `/graphs/${flowId}/executions/${runId}`;
    try {
      const response = await fetch(this.baseUrl + path);
      if (!response.ok) {
        console.warn(`GET ${path} returned non-OK response:`, response);
        throw new Error(`HTTP error ${response.status}!`);
      }
      return (await response.json()).map((result: any) => ({
        ...result,
        add_time: new Date(result.add_time),
        queue_time: result.queue_time ? new Date(result.queue_time) : undefined,
        start_time: result.start_time ? new Date(result.start_time) : undefined,
        end_time: result.end_time ? new Date(result.end_time) : undefined,
      }));
    } catch (error) {
      console.error('Error fetching execution status:', error);
      throw error;
    }
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
  input_default: Map<string, any>;
  input_nodes: Array<{ name: string, node_id: string }>;
  output_nodes: Array<{ name: string, node_id: string }>;
  metadata: {
    position: XYPosition;
    [key: string]: any;
  };
};

/* Mirror of autogpt_server/data/graph.py:Link */
export type Link = {
  source_id: string;
  sink_id: string;
  source_name: string;
  sink_name: string;
}

/* Mirror of autogpt_server/data/graph.py:GraphMeta */
export type FlowMeta = {
  graph_id: string;
  version: number;
  is_active: boolean;
  is_template: boolean;
  name: string;
  description: string;
}

/* Mirror of autogpt_server/data/graph.py:Graph */
export type Flow = FlowMeta & {
  nodes: Array<Node>;
  links: Array<Link>;
};

export type FlowCreatable = Flow | {
  graph_id?: string;
}

export type FlowCreateRequestBody = {
  template_id: string;
  template_version: number;
} | {
  graph: FlowCreatable;
}

/* Derived from autogpt_server/executor/manager.py:ExecutionManager.add_execution */
export type FlowExecuteResponse = {
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
  input_data: Map<string, any>;
  output_data: Map<string, any[]>;
  add_time: Date;
  queue_time?: Date;
  start_time?: Date;
  end_time?: Date;
};
