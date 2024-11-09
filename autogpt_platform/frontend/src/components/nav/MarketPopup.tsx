import { ButtonHTMLAttributes } from "react";
import React from "react";

interface MarketPopupProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  marketplaceUrl?: string;
}

export default function MarketPopup({
  className = "",
  marketplaceUrl = (() => {
    if (process.env.NEXT_PUBLIC_APP_ENV === "prod") {
      return "https://production-marketplace-url.com";
    } else if (process.env.NEXT_PUBLIC_APP_ENV === "dev") {
      return "https://dev-builder.agpt.co/marketplace";
    } else {
      return "http://localhost:3000/marketplace";
    }
  })(),
  children,
  ...props
}: MarketPopupProps) {
  const openMarketplacePopup = () => {
    window.open(
      marketplaceUrl,
      "popupWindow",
      "width=600,height=400,toolbar=no,menubar=no,scrollbars=no",
    );
  };

  return (
    <button onClick={openMarketplacePopup} className={className} {...props}>
      {children}
    </button>
  );
}
