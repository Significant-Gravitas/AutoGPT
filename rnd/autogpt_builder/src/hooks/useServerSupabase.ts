import { createClient } from "@/lib/supabase/server";

const useServerSupabase = () => {
    return createClient();
}

export default useServerSupabase;
