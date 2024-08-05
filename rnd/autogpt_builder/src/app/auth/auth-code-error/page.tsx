"use client";

import { useEffect, useState } from "react";

export default function AuthErrorPage() {
  const [errorType, setErrorType] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [errorDescription, setErrorDescription] = useState<string | null>(null);

  useEffect(() => {
    // This code only runs on the client side
    if (typeof window !== "undefined") {
      const hash = window.location.hash.substring(1); // Remove the leading '#'
      const params = new URLSearchParams(hash);

      setErrorType(params.get("error"));
      setErrorCode(params.get("error_code"));
      setErrorDescription(
        params.get("error_description")?.replace(/\+/g, " ") ?? null,
      ); // Replace '+' with space
    }
  }, []);

  if (!errorType && !errorCode && !errorDescription) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h1>Authentication Error</h1>
      {errorType && <p>Error Type: {errorType}</p>}
      {errorCode && <p>Error Code: {errorCode}</p>}
      {errorDescription && <p>Error Description: {errorDescription}</p>}
    </div>
  );
}
