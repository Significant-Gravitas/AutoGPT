"use client";

import { useGetV1ManagePaymentMethods } from "@/app/api/__generated__/endpoints/credits/credits";
import type { GetV1ManagePaymentMethods200 } from "@/app/api/__generated__/models/getV1ManagePaymentMethods200";

export function usePaymentMethodCard() {
  const { data: portalUrl } = useGetV1ManagePaymentMethods({
    query: {
      select: (res) => {
        const raw = res.data as GetV1ManagePaymentMethods200 | undefined;
        const url = raw && typeof raw === "object" ? raw.url : undefined;
        return typeof url === "string" ? url : undefined;
      },
    },
  });

  return {
    portalUrl,
    canManage: Boolean(portalUrl),
    onManage: () => {
      if (portalUrl) window.location.href = portalUrl;
    },
  };
}
