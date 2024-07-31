import { createServerClient } from '@/lib/supabase/server';
import { useMemo } from 'react';

const useSupabaseServer = () => {
  const supabaseClient = useMemo(() => createServerClient(), []);
  return supabaseClient;
};

export default useSupabaseServer;
