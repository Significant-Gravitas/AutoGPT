import { createServerClient } from "@/lib/supabase/server";

const getServerUser = async () => {
  const supabase = createServerClient();

  if (!supabase) {
    return { user: null, error: "Failed to create Supabase client" };
  }

  try {
    const { data, error } = await supabase.auth.getUser();
    if (error) {
      return { user: null, error: error.message };
    }
    return { user: data.user, error: null };
  } catch (error) {
    return { user: null, error: (error as Error).message };
  }
};

export default getServerUser;
