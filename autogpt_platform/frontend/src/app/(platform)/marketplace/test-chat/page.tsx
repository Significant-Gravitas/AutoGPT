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
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        setError("No Supabase session found. Please log in.");
        return;
      }
      
      setResult(`Session found!\nUser ID: ${session.user.id}\nToken: ${session.access_token.substring(0, 20)}...`);
      
      // Test BackendAPI authentication
      const api = new BackendAPI();
      const isAuth = await api.isAuthenticated();
      setResult(prev => prev + `\n\nBackendAPI authenticated: ${isAuth}`);
      
      // Test chat API
      try {
        const chatSession = await api.chat.createSession({
          system_prompt: "Test prompt"
        });
        setResult(prev => prev + `\n\nChat session created!\nSession ID: ${chatSession.id}`);
      } catch (chatError: any) {
        setResult(prev => prev + `\n\nChat API error: ${chatError.message}`);
      }
      
    } catch (err: any) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-8">
      <h1 className="text-2xl font-bold mb-4">Chat Authentication Test</h1>
      
      <div className="mb-4">
        <p className="text-sm text-gray-600">
          User: {user?.email || "Not logged in"}
        </p>
      </div>
      
      <button
        onClick={testAuth}
        disabled={loading}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
      >
        {loading ? "Testing..." : "Test Authentication"}
      </button>
      
      {error && (
        <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
          <h3 className="font-bold">Error:</h3>
          <pre className="mt-2 text-sm">{error}</pre>
        </div>
      )}
      
      {result && (
        <div className="mt-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
          <h3 className="font-bold">Result:</h3>
          <pre className="mt-2 text-sm whitespace-pre-wrap">{result}</pre>
        </div>
      )}
    </div>
  );
}