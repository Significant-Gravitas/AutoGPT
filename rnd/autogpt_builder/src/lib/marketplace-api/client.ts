import { createClient } from "../supabase/client";
import {
  AddAgentRequest,
  AgentResponse,
  ListAgentsParams,
  AgentListResponse,
  AgentDetailResponse,
  AgentWithRank,
  FeaturedAgentResponse,
  UniqueCategoriesResponse,
} from "./types";

export default class MarketplaceAPI {
  private baseUrl: string;
  private supabaseClient = createClient();

  constructor(
    baseUrl: string = process.env.NEXT_PUBLIC_AGPT_MARKETPLACE_URL ||
      "http://localhost:8001/api/v1/market",
  ) {
    this.baseUrl = baseUrl;
  }

  async checkHealth(): Promise<{ status: string }> {
    try {
      this._get("/health");
      return { status: "available" };
    } catch (error) {
      return { status: "unavailable" };
    }
  }

  async listAgents(params: ListAgentsParams = {}): Promise<AgentListResponse> {
    const queryParams = new URLSearchParams(
      Object.entries(params).filter(([_, v]) => v != null) as [
        string,
        string,
      ][],
    );
    return this._get(`/agents?${queryParams.toString()}`);
  }

  async getTopDownloadedAgents(
    page: number = 1,
    pageSize: number = 10,
  ): Promise<AgentListResponse> {
    return this._get(
      `/top-downloads/agents?page=${page}&page_size=${pageSize}`,
    );
  }

  async getFeaturedAgents(
    page: number = 1,
    pageSize: number = 10,
  ): Promise<AgentListResponse> {
    return this._get(`/featured/agents?page=${page}&page_size=${pageSize}`);
  }

  async searchAgents(
    query: string,
    page: number = 1,
    pageSize: number = 10,
    categories?: string[],
    descriptionThreshold: number = 60,
    sortBy: string = "rank",
    sortOrder: "asc" | "desc" = "desc",
  ): Promise<AgentWithRank[]> {
    const queryParams = new URLSearchParams({
      query,
      page: page.toString(),
      page_size: pageSize.toString(),
      description_threshold: descriptionThreshold.toString(),
      sort_by: sortBy,
      sort_order: sortOrder,
    });

    if (categories && categories.length > 0) {
      categories.forEach((category) =>
        queryParams.append("categories", category),
      );
    }

    return this._get(`/search?${queryParams.toString()}`);
  }

  async getAgentDetails(
    id: string,
    version?: number,
  ): Promise<AgentDetailResponse> {
    const queryParams = new URLSearchParams();
    if (version) queryParams.append("version", version.toString());
    return this._get(`/agents/${id}?${queryParams.toString()}`);
  }

  async downloadAgent(
    id: string,
    version?: number,
  ): Promise<AgentDetailResponse> {
    const queryParams = new URLSearchParams();
    if (version) queryParams.append("version", version.toString());
    return this._get(`/agents/${id}/download?${queryParams.toString()}`);
  }

  async submitAgent(
    graph: { [key: string]: any },
    author: string,
    keywords: string[],
    categories: string[],
  ): Promise<AgentResponse> {
    return this._post("/agents/submit", {
      graph,
      author,
      keywords,
      categories,
    });
  }

  async downloadAgentFile(id: string, version?: number): Promise<Blob> {
    const queryParams = new URLSearchParams();
    if (version) queryParams.append("version", version.toString());
    return this._getBlob(
      `/agents/${id}/download-file?${queryParams.toString()}`,
    );
  }

  async getAgentSubmissions(): Promise<AgentListResponse> {
    return this._get("/admin/agent/submissions");
  }

  async createAgentEntry(request: AddAgentRequest): Promise<AgentResponse> {
    return this._post("/admin/agent", request);
  }

  async approveAgentSubmission(
    agentId: string,
    version: number,
    comments: string = "",
  ): Promise<AgentResponse> {
    return this._post("/admin/agent/submissions", {
      agent_id: agentId,
      version: version,
      status: "APPROVED",
      comments: comments,
    });
  }

  async rejectAgentSubmission(
    agentId: string,
    version: number,
    comments: string = "",
  ): Promise<AgentResponse> {
    return this._post("/admin/agent/submissions", {
      agent_id: agentId,
      version: version,
      status: "REJECTED",
      comments: comments,
    });
  }

  async addFeaturedAgent(
    agentId: string,
    categories: string[],
  ): Promise<FeaturedAgentResponse> {
    const response = await this._post(`/admin/agent/featured/${agentId}`, {
      categories: categories,
    });
    return response;
  }

  async removeFeaturedAgent(
    agentId: string,
    categories: string[],
  ): Promise<FeaturedAgentResponse> {
    return this._delete(`/admin/agent/featured/${agentId}`, {
      categories: categories,
    });
  }

  async getFeaturedAgent(agentId: string): Promise<FeaturedAgentResponse> {
    return this._get(`/admin/agent/featured/${agentId}`);
  }

  async getNotFeaturedAgents(
    page: number = 1,
    pageSize: number = 10,
  ): Promise<AgentListResponse> {
    return this._get(
      `/admin/agent/not-featured?page=${page}&page_size=${pageSize}`,
    );
  }

  async getCategories(): Promise<UniqueCategoriesResponse> {
    return this._get("/admin/categories");
  }

  private async _get(path: string) {
    return this._request("GET", path);
  }

  private async _post(path: string, payload: { [key: string]: any }) {
    return this._request("POST", path, payload);
  }

  private async _delete(path: string, payload: { [key: string]: any }) {
    return this._request("DELETE", path, payload);
  }

  private async _getBlob(path: string): Promise<Blob> {
    const response = await fetch(this.baseUrl + path);
    if (!response.ok) {
      const errorData = await response.json();
      console.warn(
        `GET ${path} returned non-OK response:`,
        errorData.detail,
        response,
      );
      throw new Error(`HTTP error ${response.status}! ${errorData.detail}`);
    }
    return response.blob();
  }

  private async _request(
    method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
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
}
