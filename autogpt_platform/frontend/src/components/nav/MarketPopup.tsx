"use client";
import React from "react";
import { LuShoppingCart } from "react-icons/lu";

export function MarketPopup() {

  const openMarketplacePopup = () => {
    window.open(
      "https://dev-builder.agpt.co/marketplace",
      "popupWindow",
      "width=600,height=400,toolbar=no,menubar=no,scrollbars=no"
    );
  };
  return (
    <button
      onClick={openMarketplacePopup}
      className="flex flex-row items-center gap-2 hover:text-foreground font-medium"
    >
      <LuShoppingCart /> Marketplace
    </button>
  )
}