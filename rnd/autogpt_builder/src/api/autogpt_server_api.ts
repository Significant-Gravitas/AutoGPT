import {Block, NodeExecutionResult} from "@/types/api";

export default class AutoGPTServerAPI {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
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

  async getFlow(id: string): Promise<Flow> {
    const path = `/graphs/${id}`;
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

  async createFlow(flowCreateBody: FlowCreateBody): Promise<Flow> {
    console.debug("POST /graphs payload:", flowCreateBody);
    try {
      const response = await fetch(`${this.baseUrl}/graphs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(flowCreateBody),
      });
      if (!response.ok) {
        console.warn("POST /graphs returned non-OK response:", response);
        throw new Error(`HTTP error ${response.status}!`)
      }
      return await response.json();
    } catch (error) {
      console.error("Error storing flow:", error);
      throw error;
    }
  }

  async executeFlow(flowId: string): Promise<FlowExecuteResponse> {
    const path = `/graphs/${flowId}/execute`;
    console.debug(`POST ${path}`);
    try {
      const response = await fetch(this.baseUrl + path, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        console.warn(
          `POST /graphs/${flowId}/execute returned non-OK response:`, response
        );
        throw new Error(`HTTP error ${response.status}!`)
      }
      return await response.json();
    } catch (error) {
      console.error("Error executing flow:", error);
      throw error;
    }
  }

  async listFlowRunIDs(flowId: string): Promise<string[]> {
    const path = `/graphs/${flowId}/executions`
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
      return await response.json();
    } catch (error) {
      console.error('Error fetching execution status:', error);
      throw error;
    }
  }
}


