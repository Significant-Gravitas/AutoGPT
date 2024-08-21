import { createServerClient } from "@/lib/supabase/server";

const getServerUser = async () => {
  const supabase = createServerClient();

  if (!supabase) {
    return { user: null, error: "Failed to create Supabase client" };
  }

  try {
    const {
      data: { user },
      error,
    } = await supabase.auth.getUser();
    if (error) {
      console.error("Supabase auth error:", error);
      return { user: null, role: null, error: `Auth error: ${error.message}` };
    }
    if (!user) {
      return { user: null, role: null, error: "No user found in the response" };
    }
    const role = user.role || null;
    return { user, role, error: null };
  } catch (error) {
    console.error("Unexpected error in getServerUser:", error);
    return {
      user: null,
      role: null,
      error: `Unexpected error: ${(error as Error).message}`,
    };
  }
};

export default getServerUser;
