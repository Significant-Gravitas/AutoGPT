"use server";

import { revalidatePath } from "next/cache";
import {
  getV2GetAdminListingsHistory,
  postV2ReviewStoreSubmission,
  getV2AdminDownloadAgentFile,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";

export async function approveAgent(formData: FormData) {
  const storeListingVersionId = formData.get("id") as string;
  const comments = formData.get("comments") as string;

  await postV2ReviewStoreSubmission(storeListingVersionId, {
    store_listing_version_id: storeListingVersionId,
    is_approved: true,
    comments,
  });

  revalidatePath("/admin/marketplace");
}

export async function rejectAgent(formData: FormData) {
  const storeListingVersionId = formData.get("id") as string;
  const comments = formData.get("comments") as string;
  const internal_comments =
    (formData.get("internal_comments") as string) || undefined;

  await postV2ReviewStoreSubmission(storeListingVersionId, {
    store_listing_version_id: storeListingVersionId,
    is_approved: false,
    comments,
    internal_comments,
  });

  revalidatePath("/admin/marketplace");
}

export async function getAdminListingsWithVersions(
  status?: SubmissionStatus,
  search?: string,
  page: number = 1,
  pageSize: number = 20,
) {
  const response = await getV2GetAdminListingsHistory({
    status,
    search,
    page,
    page_size: pageSize,
  });

  return okData(response);
}

export async function downloadAsAdmin(storeListingVersion: string) {
  const response = await getV2AdminDownloadAgentFile(storeListingVersion);
  return okData(response);
}
