"use client";

import { useSupabase } from '@/components/SupabaseProvider';
import { Button } from '@/components/ui/button'
import useUser from '@/hooks/useUser';
import { useRouter } from 'next/navigation';
import { FaSpinner } from 'react-icons/fa';

export default function PrivatePage() {
  const { user, isLoading, error } = useUser()
  const { supabase } = useSupabase()
  const router = useRouter()

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-[80vh]">
        <FaSpinner className="mr-2 h-16 w-16 animate-spin" />
      </div>
    );
  }

  if (error || !user || !supabase) {
    router.push('/login')
    return null
  }

  return (
    <div>
      <p>Hello {user.email}</p>
      <Button onClick={() => supabase.auth.signOut()}>Log out</Button>
    </div>
  )
}
