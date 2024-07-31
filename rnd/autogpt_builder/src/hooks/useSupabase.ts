import { createClient } from '@/lib/supabase/client';
import { useMemo } from 'react';

const useSupabase = () => {
  const supabaseClient = useMemo(() => createClient(), []);
  return supabaseClient;
};

export default useSupabase;
