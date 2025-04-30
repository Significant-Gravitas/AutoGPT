"use server";

import { revalidatePath } from "next/cache";
import BackendApi from "@/lib/autogpt-server-api";
import {
  NotificationPreferenceDTO,
  StoreListingsWithVersionsResponse,
  StoreSubmissionsResponse,
  SubmissionStatus,
} from "@/lib/autogpt-server-api/types";

export async function approveAgent(formData: FormData) {
  const data = {
    store_listing_version_id: formData.get("id") as string,
    is_approved: true,
    comments: formData.get("comments") as string,
  };
  const api = new BackendApi();
  await api.reviewSubmissionAdmin(data.store_listing_version_id, data);

  revalidatePath("/admin/marketplace");
}

export async function rejectAgent(formData: FormData) {
  const data = {
    store_listing_version_id: formData.get("id") as string,
    is_approved: false,
    comments: formData.get("comments") as string,
    internal_comments: formData.get("internal_comments") as string,
  };
  const api = new BackendApi();
  await api.reviewSubmissionAdmin(data.store_listing_version_id, data);

  revalidatePath("/admin/marketplace");
}

export async function getAdminListingsWithVersions(
  status?: SubmissionStatus,
  search?: string,
  page: number = 1,
  pageSize: number = 20,
): Promise<StoreListingsWithVersionsResponse> {
  const data: Record<string, any> = {
    page,
    page_size: pageSize,
  };

  if (status) {
    data.status = status;
  }

  if (search) {
    data.search = search;
  }
  const api = new BackendApi();
  const response = await api.getAdminListingsWithVersions(data);
  return response;
}

export async function downloadAsAdmin(storeListingVersion: string) {
  const api = new BackendApi();
  const file = await api.downloadStoreAgentAdmin(storeListingVersion);
  return file;
}
