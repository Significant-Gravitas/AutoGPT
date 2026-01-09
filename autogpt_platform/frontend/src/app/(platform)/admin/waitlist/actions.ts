"use server";

import { revalidatePath } from "next/cache";
import BackendAPI from "@/lib/autogpt-server-api";
import type {
  WaitlistAdminResponse,
  WaitlistAdminListResponse,
  WaitlistSignupListResponse,
  WaitlistCreateRequest,
  WaitlistUpdateRequest,
} from "@/lib/autogpt-server-api/types";

export async function getWaitlistsAdmin(): Promise<WaitlistAdminListResponse> {
  const api = new BackendAPI();
  const response = await api.getWaitlistsAdmin();
  return response;
}

export async function getWaitlistAdmin(
  waitlistId: string,
): Promise<WaitlistAdminResponse> {
  const api = new BackendAPI();
  const response = await api.getWaitlistAdmin(waitlistId);
  return response;
}

export async function createWaitlist(
  data: WaitlistCreateRequest,
): Promise<WaitlistAdminResponse> {
  const api = new BackendAPI();
  const response = await api.createWaitlist(data);
  revalidatePath("/admin/waitlist");
  return response;
}

export async function updateWaitlist(
  waitlistId: string,
  data: WaitlistUpdateRequest,
): Promise<WaitlistAdminResponse> {
  const api = new BackendAPI();
  const response = await api.updateWaitlist(waitlistId, data);
  revalidatePath("/admin/waitlist");
  return response;
}

export async function deleteWaitlist(waitlistId: string): Promise<void> {
  const api = new BackendAPI();
  await api.deleteWaitlist(waitlistId);
  revalidatePath("/admin/waitlist");
}

export async function getWaitlistSignups(
  waitlistId: string,
): Promise<WaitlistSignupListResponse> {
  const api = new BackendAPI();
  const response = await api.getWaitlistSignups(waitlistId);
  return response;
}

export async function linkWaitlistToListing(
  waitlistId: string,
  storeListingId: string,
): Promise<WaitlistAdminResponse> {
  const api = new BackendAPI();
  const response = await api.linkWaitlistToListing(waitlistId, storeListingId);
  revalidatePath("/admin/waitlist");
  return response;
}
