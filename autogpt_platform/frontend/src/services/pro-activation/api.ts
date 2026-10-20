import {
  getSubscriptionsGetCurrentProActivation,
  getSubscriptionsGetProActivation,
  postSubscriptionsConfirmProActivation,
  postSubscriptionsPreviewProActivation,
} from "@/app/api/__generated__/endpoints/subscriptions/subscriptions";
import type { ActivationResponse } from "@/app/api/__generated__/models/activationResponse";

function response<T extends { status: number; data: unknown }>(result: T) {
  if (result.status !== 200)
    throw new Error("Activation request could not be completed");
  return result.data as ActivationResponse;
}

export async function previewActivation(returnTo: string) {
  return response(
    await postSubscriptionsPreviewProActivation({ return_to: returnTo }),
  );
}

export async function confirmActivation(id: string, token: string) {
  return response(
    await postSubscriptionsConfirmProActivation(id, {
      confirmed: true,
      terms_token: token,
    }),
  );
}

export async function retrieveActivation(id?: string | null) {
  return response(
    await (id
      ? getSubscriptionsGetProActivation(id)
      : getSubscriptionsGetCurrentProActivation()),
  );
}

export function statusCode(error: unknown) {
  return error && typeof error === "object" && "status" in error
    ? error.status
    : undefined;
}
