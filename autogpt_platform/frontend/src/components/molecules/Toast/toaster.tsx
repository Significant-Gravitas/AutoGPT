"use client";

import { Toaster as HotToaster } from "react-hot-toast";

export function Toaster() {
  return (
    <HotToaster
      position="bottom-center"
      gutter={8}
      containerClassName="custom__toast"
    />
  );
}
