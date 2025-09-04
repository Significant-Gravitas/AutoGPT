"use client";

import React, { useState } from "react";
import BackendAPI from "@/lib/autogpt-server-api";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

export default function TestChatPage() {
  const [result, setResult] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const { supabase, user } = useSupabase();

  const testAuth = async () => {
    setLoading(true);
    setError("");
    setResult("");

    try {
      // Test Supabase session
      if (!supabase) {
        setError("Supabase client not initialized");
        return;
      }

      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session) {
        setError("No Supabase session found. Please log in.");
        return;
      }

      setResult(
        `Session found!\nUser ID: ${session.user.id}\nToken: ${session.access_token.substring(0, 20)}...`,
      );

      // Test BackendAPI authentication
      const api = new BackendAPI();
      const isAuth = await api.isAuthenticated();
      setResult((prev) => prev + `\n\nBackendAPI authenticated: ${isAuth}`);

      // Test chat API
      try {
        const chatSession = await api.chat.createSession({
          system_prompt: "Test prompt",
        });
        setResult(
          (prev) =>
            prev + `\n\nChat session created!\nSession ID: ${chatSession.id}`,
        );
      } catch (chatError: any) {
        setResult((prev) => prev + `\n\nChat API error: ${chatError.message}`);
      }
    } catch (err: any) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-8">
      <h1 className="mb-4 text-2xl font-bold">Chat Authentication Test</h1>

      <div className="mb-4">
        <p className="text-sm text-gray-600">
          User: {user?.email || "Not logged in"}
        </p>
      </div>

      <button
        onClick={testAuth}
        disabled={loading}
        className="rounded bg-blue-500 px-4 py-2 text-white hover:bg-blue-600 disabled:bg-gray-400"
      >
        {loading ? "Testing..." : "Test Authentication"}
      </button>

      {error && (
        <div className="mt-4 rounded border border-red-400 bg-red-100 p-4 text-red-700">
          <h3 className="font-bold">Error:</h3>
          <pre className="mt-2 text-sm">{error}</pre>
        </div>
      )}

      {result && (
        <div className="mt-4 rounded border border-green-400 bg-green-100 p-4 text-green-700">
          <h3 className="font-bold">Result:</h3>
          <pre className="mt-2 whitespace-pre-wrap text-sm">{result}</pre>
        </div>
      )}
    </div>
  );
}
