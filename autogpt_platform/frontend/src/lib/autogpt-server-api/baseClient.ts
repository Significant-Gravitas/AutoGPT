import { SupabaseClient } from "@supabase/supabase-js";
import {
  AnalyticsMetrics,
  AnalyticsDetails,
  APIKeyCredentials,
  Block,
  CredentialsMetaResponse,
  Graph,
  GraphCreatable,
  GraphUpdateable,
  GraphMeta,
  GraphMetaWithRuns,
  GraphExecuteResponse,
  ExecutionMeta,
  NodeExecutionResult,
  OAuth2Credentials,
  User,
} from "./types";

export default class BaseAutoGPTServerAPI {
  private baseUrl: string;
  private wsUrl: string;
  private webSocket: WebSocket | null = null;
  private wsConnecting: Promise<void> | null = null;
  private wsMessageHandlers: Record<string, Set<(data: any) => void>> = {};
  private supabaseClient: SupabaseClient | null = null;

  constructor(
    baseUrl: string = process.env.NEXT_PUBLIC_AGPT_SERVER_URL ||
      "http://localhost:8006/api",
    wsUrl: string = process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL ||
      "ws://localhost:8001/ws",
    supabaseClient: SupabaseClient | null = null,
  ) {
    this.baseUrl = baseUrl;
    this.wsUrl = wsUrl;
    this.supabaseClient = supabaseClient;
  }

  async isAuthenticated(): Promise<boolean> {
    if (!this.supabaseClient) return false;
    const {
      data: { session },
    } = await this.supabaseClient?.auth.getSession();
    return session != null;
  }

  createUser(): Promise<User> {
    return this._request("POST", "/auth/user", {});
  }

  getUserCredit(): Promise<{ credits: number }> {
    return this._get(`/credits`);
  }

  getBlocks(): Promise<Block[]> {
    return this._get("/blocks");
  }

  listGraphs(): Promise<GraphMeta[]> {
    return this._get(`/graphs`);
  }

  async listGraphsWithRuns(): Promise<GraphMetaWithRuns[]> {
    let graphs = await this._get(`/graphs?with_runs=true`);
    return graphs.map(parseGraphMetaWithRuns);
  }

  listTemplates(): Promise<GraphMeta[]> {
    return this._get("/templates");
  }

  getGraph(
    id: string,
    version?: number,
    hide_credentials?: boolean,
  ): Promise<Graph> {
    let query: Record<string, any> = {};
    if (version !== undefined) {
      query["version"] = version;
    }
    if (hide_credentials !== undefined) {
      query["hide_credentials"] = hide_credentials;
    }
    return this._get(`/graphs/${id}`, query);
  }

  getTemplate(id: string, version?: number): Promise<Graph> {
    const query = version !== undefined ? `?version=${version}` : "";
    return this._get(`/templates/${id}` + query);
  }

  getGraphAllVersions(id: string): Promise<Graph[]> {
    return this._get(`/graphs/${id}/versions`);
  }

  getTemplateAllVersions(id: string): Promise<Graph[]> {
    return this._get(`/templates/${id}/versions`);
  }

  createGraph(graphCreateBody: GraphCreatable): Promise<Graph>;
  createGraph(fromTemplateID: string, templateVersion: number): Promise<Graph>;
  createGraph(
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

  createTemplate(templateCreateBody: GraphCreatable): Promise<Graph> {
    const requestBody: GraphCreateRequestBody = { graph: templateCreateBody };
    return this._request("POST", "/templates", requestBody);
  }

  updateGraph(id: string, graph: GraphUpdateable): Promise<Graph> {
    return this._request("PUT", `/graphs/${id}`, graph);
  }

  updateTemplate(id: string, template: GraphUpdateable): Promise<Graph> {
    return this._request("PUT", `/templates/${id}`, template);
  }

  deleteGraph(id: string): Promise<void> {
    return this._request("DELETE", `/graphs/${id}`);
  }

  setGraphActiveVersion(id: string, version: number): Promise<Graph> {
    return this._request("PUT", `/graphs/${id}/versions/active`, {
      active_graph_version: version,
    });
  }

  executeGraph(
    id: string,
    inputData: { [key: string]: any } = {},
  ): Promise<GraphExecuteResponse> {
    return this._request("POST", `/graphs/${id}/execute`, inputData);
  }

  listGraphRunIDs(graphID: string, graphVersion?: number): Promise<string[]> {
    const query =
      graphVersion !== undefined ? `?graph_version=${graphVersion}` : "";
    return this._get(`/graphs/${graphID}/executions` + query);
  }

  async getGraphExecutionInfo(
    graphID: string,
    runID: string,
  ): Promise<NodeExecutionResult[]> {
    return (await this._get(`/graphs/${graphID}/executions/${runID}`)).map(
      parseNodeExecutionResultTimestamps,
    );
  }

  async stopGraphExecution(
    graphID: string,
    runID: string,
  ): Promise<NodeExecutionResult[]> {
    return (
      await this._request("POST", `/graphs/${graphID}/executions/${runID}/stop`)
    ).map(parseNodeExecutionResultTimestamps);
  }

  oAuthLogin(
    provider: string,
    scopes?: string[],
  ): Promise<{ login_url: string; state_token: string }> {
    const query = scopes ? { scopes: scopes.join(",") } : undefined;
    return this._get(`/integrations/${provider}/login`, query);
  }

  oAuthCallback(
    provider: string,
    code: string,
    state_token: string,
  ): Promise<CredentialsMetaResponse> {
    return this._request("POST", `/integrations/${provider}/callback`, {
      code,
      state_token,
    });
  }

  createAPIKeyCredentials(
    credentials: Omit<APIKeyCredentials, "id" | "type">,
  ): Promise<APIKeyCredentials> {
    return this._request(
      "POST",
      `/integrations/${credentials.provider}/credentials`,
      credentials,
    );
  }

  listCredentials(provider: string): Promise<CredentialsMetaResponse[]> {
    return this._get(`/integrations/${provider}/credentials`);
  }

  getCredentials(
    provider: string,
    id: string,
  ): Promise<APIKeyCredentials | OAuth2Credentials> {
    return this._get(`/integrations/${provider}/credentials/${id}`);
  }

  deleteCredentials(provider: string, id: string): Promise<void> {
    return this._request(
      "DELETE",
      `/integrations/${provider}/credentials/${id}`,
    );
  }

  logMetric(metric: AnalyticsMetrics) {
    return this._request("POST", "/analytics/log_raw_metric", metric);
  }

  logAnalytic(analytic: AnalyticsDetails) {
    return this._request("POST", "/analytics/log_raw_analytics", analytic);
  }

  private async _get(path: string, query?: Record<string, any>) {
    return this._request("GET", path, query);
  }

  private async _request(
    method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
    path: string,
    payload?: Record<string, any>,
  ) {
    if (method != "GET") {
      console.debug(`${method} ${path} payload:`, payload);
    }

    const token =
      (await this.supabaseClient?.auth.getSession())?.data.session
        ?.access_token || "";

    let url = this.baseUrl + path;
    if (method === "GET" && payload) {
      // For GET requests, use payload as query
      const queryParams = new URLSearchParams(payload);
      url += `?${queryParams.toString()}`;
    }

    const hasRequestBody = method !== "GET" && payload !== undefined;
    const response = await fetch(url, {
      method,
      headers: hasRequestBody
        ? {
            "Content-Type": "application/json",
            Authorization: token ? `Bearer ${token}` : "",
          }
        : {
            Authorization: token ? `Bearer ${token}` : "",
          },
      body: hasRequestBody ? JSON.stringify(payload) : undefined,
    });
    const response_data = await response.json();

    if (!response.ok) {
      console.warn(
        `${method} ${path} returned non-OK response:`,
        response_data.detail,
        response,
      );

      if (
        response.status === 403 &&
        response_data.detail === "Not authenticated" &&
        window // Browser environment only: redirect to login page.
      ) {
        window.location.href = "/login";
      }

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
          console.debug("WebSocket connection established");
          resolve();
        };

        this.webSocket.onclose = (event) => {
          console.debug("WebSocket connection closed", event);
          this.webSocket = null;
        };

        this.webSocket.onerror = (error) => {
          console.error("WebSocket error:", error);
          reject(error);
        };

        this.webSocket.onmessage = (event) => {
          const message: WebsocketMessage = JSON.parse(event.data);
          if (message.method == "execution_event") {
            message.data = parseNodeExecutionResultTimestamps(message.data);
          }
          this.wsMessageHandlers[message.method]?.forEach((handler) =>
            handler(message.data),
          );
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
  ): () => void {
    this.wsMessageHandlers[method] ??= new Set();
    this.wsMessageHandlers[method].add(handler);

    // Return detacher
    return () => this.wsMessageHandlers[method].delete(handler);
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

type WebsocketMessage = {
  [M in keyof WebsocketMessageTypeMap]: {
    method: M;
    data: WebsocketMessageTypeMap[M];
  };
}[keyof WebsocketMessageTypeMap];

/* *** HELPER FUNCTIONS *** */

function parseNodeExecutionResultTimestamps(result: any): NodeExecutionResult {
  return {
    ...result,
    add_time: new Date(result.add_time),
    queue_time: result.queue_time ? new Date(result.queue_time) : undefined,
    start_time: result.start_time ? new Date(result.start_time) : undefined,
    end_time: result.end_time ? new Date(result.end_time) : undefined,
  };
}

function parseGraphMetaWithRuns(result: any): GraphMetaWithRuns {
  return {
    ...result,
    executions: result.executions
      ? result.executions.map(parseExecutionMetaTimestamps)
      : [],
  };
}

function parseExecutionMetaTimestamps(result: any): ExecutionMeta {
  let status: "running" | "waiting" | "success" | "failed" = "success";
  if (result.status === "FAILED") {
    status = "failed";
  } else if (["QUEUED", "RUNNING"].includes(result.status)) {
    status = "running";
  } else if (result.status === "INCOMPLETE") {
    status = "waiting";
  }

  return {
    ...result,
    status,
    started_at: new Date(result.started_at).getTime(),
    ended_at: result.ended_at ? new Date(result.ended_at).getTime() : undefined,
  };
}
