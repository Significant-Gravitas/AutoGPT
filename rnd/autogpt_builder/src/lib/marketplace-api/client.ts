import {
    AddAgentRequest,
    Agent,
    AgentList,
    AgentDetail

} from "./types"

export default class MarketplaceAPI {
    private baseUrl: string;
    private messageHandlers: { [key: string]: (data: any) => void } = {};

    constructor(
        baseUrl: string = process.env.AGPT_SERVER_URL || "http://localhost:8000/api"
    ) {
        this.baseUrl = baseUrl;
    }


    async listAgents(): Promise<AgentList> {
        return this._get("/agents")
    }

    async getAgent(id: string): Promise<AgentDetail> {
        return this._get(`/agents/${id}`);
    }

    async addAgent(agent: AddAgentRequest): Promise<Agent> {
        return this._post("/agents", agent);
    }

    async downloadAgent(id: string): Promise<Blob> {
        return this._get(`/agents/${id}/download`);
    }

    private async _get(path: string) {
        return this._request("GET", path);
    }

    private async _post(path: string, payload: { [key: string]: any }) {
        return this._request("POST", path, payload);
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
}