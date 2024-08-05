"use client";

import { createClient } from "@/lib/supabase/client";
import { SupabaseClient } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";

type SupabaseContextType = {
  supabase: SupabaseClient | null;
  isLoading: boolean;
};

const Context = createContext<SupabaseContextType | undefined>(undefined);

export default function SupabaseProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [supabase, setSupabase] = useState<SupabaseClient | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const initializeSupabase = async () => {
      setIsLoading(true);
      const client = createClient();
      setSupabase(client);
      setIsLoading(false);

      if (client) {
        const {
          data: { subscription },
        } = client.auth.onAuthStateChange(() => {
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
