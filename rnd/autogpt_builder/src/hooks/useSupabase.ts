import { createClient } from "@/lib/supabase/client";

const useSupabase = () => {
    return createClient();
}

export default useSupabase;
