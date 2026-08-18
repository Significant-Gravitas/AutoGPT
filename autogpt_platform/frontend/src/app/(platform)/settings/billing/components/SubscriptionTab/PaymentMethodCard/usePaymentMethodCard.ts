"use client";

import { useState } from "react";

import { useGetV1ManagePaymentMethods } from "@/app/api/__generated__/endpoints/credits/credits";
import type { GetV1ManagePaymentMethods200 } from "@/app/api/__generated__/models/getV1ManagePaymentMethods200";

export function usePaymentMethodCard() {
  const [isOpening, setIsOpening] = useState(false);

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
    isOpening,
    onManage: () => {
      if (!portalUrl) return;
      // Surface a loading state for the brief gap between click and the
      // browser starting the cross-origin navigation to Stripe — without
      // it the button looks unresponsive on slower networks.
      setIsOpening(true);
      window.location.href = portalUrl;
    },
  };
}
