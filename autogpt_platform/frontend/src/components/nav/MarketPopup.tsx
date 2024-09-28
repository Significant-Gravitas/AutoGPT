"use client";
import React from "react";
import { LuShoppingCart } from "react-icons/lu";
import { ButtonHTMLAttributes } from "react";

interface MarketPopupProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  marketplaceUrl?: string;
}

export function MarketPopup({
  className = "",
  marketplaceUrl = "http://platform.agpt.co/marketplace",
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
      <LuShoppingCart /> Marketplace
    </button>
  );
}
