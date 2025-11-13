"use server";

import { postV2AddMarketplaceAgent } from "@/app/api/__generated__/endpoints/library/library";
import { getV1OnboardingState } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { resolveResponse } from "@/app/api/helpers";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";

export async function finishOnboarding() {
  const onboarding = await resolveResponse(getV1OnboardingState());

  const listingId = onboarding?.selectedStoreListingVersionId;
  if (listingId) {
    const data = await resolveResponse(
      postV2AddMarketplaceAgent({
        store_listing_version_id: listingId,
      }),
    );
    revalidatePath(`/library/agents/${data.id}`, "layout");
    redirect(`/library/agents/${data.id}`);
  } else {
    revalidatePath("/library", "layout");
    redirect("/library");
  }
}
