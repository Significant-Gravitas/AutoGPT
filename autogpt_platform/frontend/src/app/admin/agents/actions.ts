"use server";

import { revalidatePath } from "next/cache";
import BackendApi from "@/lib/autogpt-server-api";
import {
  NotificationPreferenceDTO,
  StoreSubmissionsResponse,
  SubmissionStatus,
} from "@/lib/autogpt-server-api/types";

export async function approveAgent(formData: FormData) {
  console.log("approving agent", formData);
  const data = {
    store_listing_version_id: formData.get("id") as string,
    is_approved: true,
    comments: formData.get("comments") as string,
  };
  const api = new BackendApi();
  await api.reviewSubmissionAdmin(data.store_listing_version_id, data);

  revalidatePath("/admin/agents");
}

export async function rejectAgent(formData: FormData) {
  console.log("rejecting agent", formData);
  const data = {
    store_listing_version_id: formData.get("id") as string,
    is_approved: false,
    comments: formData.get("comments") as string,
    internal_comments: formData.get("internal_comments") as string,
  };
  const api = new BackendApi();
  await api.reviewSubmissionAdmin(data.store_listing_version_id, data);

  revalidatePath("/admin/agents");
}

export async function getPendingAgents(): Promise<StoreSubmissionsResponse> {
  console.log("getting pending agents");
  const api = new BackendApi();
  const submissions = await api.getPendingSubmissionsAdmin();

  return submissions;
}

export async function getSubmissions(): Promise<StoreSubmissionsResponse> {
  console.log("getting submissions");
  const data = {
    // status: formData.get("status") as SubmissionStatus | undefined,
    // search: formData.get("search") as string | undefined,
    page: 1,
    page_size: 20,
  };
  const api = new BackendApi();
  const submission = await api.getSubmissionsAdmin(data);
  return submission;
}
