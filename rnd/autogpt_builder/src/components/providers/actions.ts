"use server";

import getServerUser from "@/hooks/getServerUser";
import AutoGPTServerAPIServerSide from "@/lib/autogpt-server-api/clientServer";

export default async function logPageViewAction(page: string, data: any) {
  console.debug("logPageViewAction", page, data);
  const apiClient = new AutoGPTServerAPIServerSide();
  await apiClient.logPageView({ page, data });
}
