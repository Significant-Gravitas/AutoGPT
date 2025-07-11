"use server";
import BackendAPI from "@/lib/autogpt-server-api";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";

export async function finishOnboarding() {
  const api = new BackendAPI();
  const onboarding = await api.getUserOnboarding();
  const listingId = onboarding?.selectedStoreListingVersionId;
  if (listingId) {
    const libraryAgent = await api.addMarketplaceAgentToLibrary(listingId);
    revalidatePath(`/library/agents/${libraryAgent.id}`, "layout");
    redirect(`/library/agents/${libraryAgent.id}`);
  } else {
    revalidatePath("/library", "layout");
    redirect("/library");
  }
}
