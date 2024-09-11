import AutoGPTServerAPI from "@/lib/autogpt-server-api/client";

export default function logPageViewAction(page: string, data: any) {
  const apiClient = new AutoGPTServerAPI();
  apiClient.logPageView({ page, data });
}
