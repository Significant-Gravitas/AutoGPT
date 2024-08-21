import { createClient } from "../supabase/client";
import {
  Block,
  Graph,
  GraphCreatable,
  GraphUpdateable,
  GraphMeta,
  GraphExecuteResponse,
  NodeExecutionResult,
  User,
} from "./types";

export default class AutoGPTServerAPI {
  private baseUrl: string;
  private wsUrl: string;
  private webSocket: WebSocket | null = null;
  private wsConnecting: Promise<void> | null = null;
  private wsMessageHandlers: { [key: string]: (data: any) => void } = {};
  private supabaseClient = createClient();

  constructor(
    baseUrl: string = process.env.NEXT_PUBLIC_AGPT_SERVER_URL ||
      "http://localhost:8000/api",
  ) {
    this.baseUrl = baseUrl;
    this.wsUrl = `ws://${new URL(this.baseUrl).host}/ws`;
  }

  async createUser(): Promise<User> {
    return this._request("POST", "/auth/user", {});
  }

  async getBlocks(): Promise<Block[]> {
    return await this._get("/blocks");
  }

  async listGraphs(): Promise<GraphMeta[]> {
    return this._get("/graphs");
  }

  async listTemplates(): Promise<GraphMeta[]> {
    return this._get("/templates");
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
  async createGraph(
    fromTemplateID: string,
    templateVersion: number,
  ): Promise<Graph>;
  async createGraph(
    graphOrTemplateID: GraphCreatable | string,
    templateVersion?: number,
  ): Promise<Graph> {
    let requestBody: GraphCreateRequestBody;

    if (typeof graphOrTemplateID == "string") {
      if (templateVersion == undefined) {
        throw new Error("templateVersion not specified");
      }
      requestBody = {
        template_id: graphOrTemplateID,
        template_version: templateVersion,
      };
    } else {
      requestBody = { graph: graphOrTemplateID };
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
    return this._request("PUT", `/graphs/${id}/versions/active`, {
      active_graph_version: version,
    });
  }

  async executeGraph(
    id: string,
    inputData: { [key: string]: any } = {},
  ): Promise<GraphExecuteResponse> {
    return this._request("POST", `/graphs/${id}/execute`, inputData);
  }

  async listGraphRunIDs(
    graphID: string,
    graphVersion?: number,
  ): Promise<string[]> {
    const query =
      graphVersion !== undefined ? `?graph_version=${graphVersion}` : "";
    return this._get(`/graphs/${graphID}/executions` + query);
  }

  async getGraphExecutionInfo(
    graphID: string,
    runID: string,
  ): Promise<NodeExecutionResult[]> {
    return (await this._get(`/graphs/${graphID}/executions/${runID}`)).map(
      (result: any) => ({
        ...result,
        add_time: new Date(result.add_time),
        queue_time: result.queue_time ? new Date(result.queue_time) : undefined,
        start_time: result.start_time ? new Date(result.start_time) : undefined,
        end_time: result.end_time ? new Date(result.end_time) : undefined,
      }),
    );
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

    const token =
      (await this.supabaseClient?.auth.getSession())?.data.session
        ?.access_token || "";

    const response = await fetch(this.baseUrl + path, {
      method,
      headers:
        method != "GET"
          ? {
              "Content-Type": "application/json",
              Authorization: token ? `Bearer ${token}` : "",
            }
          : {
              Authorization: token ? `Bearer ${token}` : "",
            },
      body: JSON.stringify(payload),
    });
    const response_data = await response.json();

    if (!response.ok) {
      console.warn(
        `${method} ${path} returned non-OK response:`,
        response_data.detail,
        response,
      );
      throw new Error(`HTTP error ${response.status}! ${response_data.detail}`);
    }
    return response_data;
  }

  async connectWebSocket(): Promise<void> {
    this.wsConnecting ??= new Promise(async (resolve, reject) => {
      try {
        const token =
          (await this.supabaseClient?.auth.getSession())?.data.session
            ?.access_token || "";

        const wsUrlWithToken = `${this.wsUrl}?token=${token}`;
        this.webSocket = new WebSocket(wsUrlWithToken);

        this.webSocket.onopen = () => {
          console.log("WebSocket connection established");
          resolve();
        };

        this.webSocket.onclose = (event) => {
          console.log("WebSocket connection closed", event);
          this.webSocket = null;
        };

        this.webSocket.onerror = (error) => {
          console.error("WebSocket error:", error);
          reject(error);
        };

        this.webSocket.onmessage = (event) => {
          const message = JSON.parse(event.data);
          if (this.wsMessageHandlers[message.method]) {
            this.wsMessageHandlers[message.method](message.data);
          }
        };
      } catch (error) {
        console.error("Error connecting to WebSocket:", error);
        reject(error);
      }
    });
    return this.wsConnecting;
  }

  disconnectWebSocket() {
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.close();
    }
  }

  sendWebSocketMessage<M extends keyof WebsocketMessageTypeMap>(
    method: M,
    data: WebsocketMessageTypeMap[M],
    callCount = 0,
  ) {
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(JSON.stringify({ method, data }));
    } else {
      this.connectWebSocket().then(() => {
        callCount == 0
          ? this.sendWebSocketMessage(method, data, callCount + 1)
          : setTimeout(
              () => {
                this.sendWebSocketMessage(method, data, callCount + 1);
              },
              2 ** (callCount - 1) * 1000,
            );
      });
    }
  }

  onWebSocketMessage<M extends keyof WebsocketMessageTypeMap>(
    method: M,
    handler: (data: WebsocketMessageTypeMap[M]) => void,
  ) {
    this.wsMessageHandlers[method] = handler;
  }

  subscribeToExecution(graphId: string) {
    this.sendWebSocketMessage("subscribe", { graph_id: graphId });
  }
}

/* *** UTILITY TYPES *** */

type GraphCreateRequestBody =
  | {
      template_id: string;
      template_version: number;
    }
  | {
      graph: GraphCreatable;
    };

type WebsocketMessageTypeMap = {
  subscribe: { graph_id: string };
  execution_event: NodeExecutionResult;
};
