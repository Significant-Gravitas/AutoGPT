/**
 * Authentication Context and Provider for Supabase Auth
 *
 * This module provides authentication state management using Supabase.
 *
 * Usage:
 * 1. Wrap your app with AuthProvider:
 *    ```tsx
 *    <AuthProvider>
 *      <App />
 *    </AuthProvider>
 *    ```
 *
 * 2. Access auth state in child components using useAuth hook:
 *    ```tsx
 *    const MyComponent = () => {
 *      const { user, session, role } = useAuth();
 *
 *      if (!user) return <div>Please log in</div>;
 *
 *      return <div>Welcome {user.email}</div>;
 *    };
 *    ```
 *
 * The context provides:
 * - user: Current authenticated User object or null
 * - session: Current Session object or null
 * - role: User's role string or null
 */

import { User, Session } from "@supabase/supabase-js";
import { createContext, useContext } from "react";
import { createServerClient } from "@/lib/supabase/server";

interface AuthContextType {
  user: User | null;
  session: Session | null;
  role: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export async function AuthProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = createServerClient();

  if (!supabase) {
    console.error("Could not create Supabase client");
    return null;
  }

  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession();

  if (userError) {
    console.error("Error fetching user:", userError);
  }

  if (sessionError) {
    console.error("Error fetching session:", sessionError);
  }

  return (
    <AuthContext.Provider
      value={{
        user: user || null,
        session: session || null,
        role: user?.role || null,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return context;
};
