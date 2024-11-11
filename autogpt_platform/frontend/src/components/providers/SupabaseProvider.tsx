"use client";

import { createClient } from "@/lib/supabase/client";
import { SupabaseClient } from "@supabase/supabase-js";
import { Session } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";

type SupabaseContextType = {
  supabase: SupabaseClient | null;
  isLoading: boolean;
};

const Context = createContext<SupabaseContextType | undefined>(undefined);

export default function SupabaseProvider({
  children,
  initialSession,
}: {
  children: React.ReactNode;
  initialSession: Session | null;
}) {
  const [session, setSession] = useState<Session | null>(initialSession);
  const [supabase, setSupabase] = useState<SupabaseClient | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const initializeSupabase = async () => {
      setIsLoading(true);
      const client = createClient();
      const api = new AutoGPTServerAPI();
      setSupabase(client);
      setIsLoading(false);

      if (client) {
        const {
          data: { subscription },
        } = client.auth.onAuthStateChange((event, session) => {
          setSession(session);
          if (event === "SIGNED_IN") {
            api.createUser();
          }
          router.refresh();
        });

        return () => {
          subscription.unsubscribe();
        };
      }
    };

    initializeSupabase();
  }, [router]);

  return (
    <Context.Provider value={{ supabase, isLoading }}>
      {children}
    </Context.Provider>
  );
}

export const useSupabase = () => {
  const context = useContext(Context);
  if (context === undefined) {
    throw new Error("useSupabase must be used inside SupabaseProvider");
  }
  return context;
};
