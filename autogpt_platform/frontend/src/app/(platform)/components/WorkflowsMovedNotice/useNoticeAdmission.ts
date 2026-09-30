"use client";

import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { hasCompetingDialog } from "./helpers";

export function useNoticeAdmission(
  isReadyToShow: boolean,
  userID: string | null,
) {
  const pathname = usePathname();
  const [admission, setAdmission] = useState<{
    pathname: string | null;
    userID: string | null;
    allowed: boolean;
  } | null>(null);
  const isCurrentVisit =
    admission?.pathname === pathname && admission?.userID === userID;
  const isDeferred = isCurrentVisit && admission?.allowed === false;

  useEffect(() => setAdmission(null), [pathname, userID]);

  useEffect(() => {
    if (!isReadyToShow || isDeferred) return;

    const observer = new MutationObserver(() => {
      if (hasCompetingDialog()) {
        setAdmission({ pathname, userID, allowed: false });
      }
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["role", "hidden", "data-state"],
    });
    const frame = requestAnimationFrame(() => {
      setAdmission({ pathname, userID, allowed: !hasCompetingDialog() });
    });
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [pathname, userID, isReadyToShow, isDeferred]);

  return isCurrentVisit && admission?.allowed === true;
}
