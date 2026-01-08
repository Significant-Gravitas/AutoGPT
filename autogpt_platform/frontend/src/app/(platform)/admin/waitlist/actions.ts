"use server";

import { revalidatePath } from "next/cache";
import BackendAPI from "@/lib/autogpt-server-api";

export type WaitlistAdminResponse = {
  id: string;
  createdAt: string;
  updatedAt: string;
  slug: string;
  name: string;
  subHeading: string;
  description: string;
  categories: string[];
  imageUrls: string[];
  videoUrl: string | null;
  agentOutputDemoUrl: string | null;
  status: string;
  votes: number;
  signupCount: number;
  storeListingId: string | null;
  owningUserId: string;
};

export type WaitlistAdminListResponse = {
  waitlists: WaitlistAdminResponse[];
  totalCount: number;
};

export type WaitlistSignup = {
  type: "user" | "email";
  userId: string | null;
  email: string | null;
  username: string | null;
};

export type WaitlistSignupListResponse = {
  waitlistId: string;
  signups: WaitlistSignup[];
  totalCount: number;
};

export type WaitlistCreateRequest = {
  name: string;
  slug: string;
  subHeading: string;
  description: string;
  categories?: string[];
  imageUrls?: string[];
  videoUrl?: string | null;
  agentOutputDemoUrl?: string | null;
};

export type WaitlistUpdateRequest = {
  name?: string;
  slug?: string;
  subHeading?: string;
  description?: string;
  categories?: string[];
  imageUrls?: string[];
  videoUrl?: string | null;
  agentOutputDemoUrl?: string | null;
  status?: string;
  storeListingId?: string | null;
};

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
