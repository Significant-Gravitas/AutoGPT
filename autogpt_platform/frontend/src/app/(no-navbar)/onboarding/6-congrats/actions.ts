"use server";
import BackendAPI from "@/lib/autogpt-server-api";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";

export async function finishOnboarding() {
  const api = new BackendAPI();
  const onboarding = await api.getUserOnboarding();
  revalidatePath("/library", "layout");
  redirect("/library");
}
