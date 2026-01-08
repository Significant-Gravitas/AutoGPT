"use server";

import { revalidatePath } from "next/cache";
import BackendApi from "@/lib/autogpt-server-api";
import type {
  OAuthAppsListResponse,
  OAuthApplicationCreationResult,
  OAuthApplication,
  CreateOAuthAppRequest,
  UpdateOAuthAppRequest,
  RegenerateSecretResponse,
} from "@/lib/autogpt-server-api/types";

export async function getOAuthApps(
  page: number = 1,
  pageSize: number = 20,
  search?: string,
): Promise<OAuthAppsListResponse> {
  const api = new BackendApi();
  return api.getOAuthApps({
    page,
    page_size: pageSize,
    search,
  });
}

export async function getOAuthApp(appId: string): Promise<OAuthApplication> {
  const api = new BackendApi();
  return api.getOAuthApp(appId);
}

export async function createOAuthApp(
  request: CreateOAuthAppRequest,
): Promise<OAuthApplicationCreationResult> {
  const api = new BackendApi();
  const result = await api.createOAuthApp(request);
  revalidatePath("/admin/oauth");
  return result;
}

export async function updateOAuthApp(
  appId: string,
  request: UpdateOAuthAppRequest,
): Promise<OAuthApplication> {
  const api = new BackendApi();
  const result = await api.updateOAuthApp(appId, request);
  revalidatePath("/admin/oauth");
  return result;
}

export async function deleteOAuthApp(appId: string): Promise<void> {
  const api = new BackendApi();
  await api.deleteOAuthApp(appId);
  revalidatePath("/admin/oauth");
}

export async function regenerateOAuthSecret(
  appId: string,
): Promise<RegenerateSecretResponse> {
  const api = new BackendApi();
  const result = await api.regenerateOAuthSecret(appId);
  revalidatePath("/admin/oauth");
  return result;
}
