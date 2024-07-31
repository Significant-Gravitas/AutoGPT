import useServerUser from '@/hooks/useServerUser'
import { redirect } from 'next/navigation'

export default async function PrivatePage() {
  const { user, error } = await useServerUser()

  if (error || !user) {
    redirect('/login')
  }

  return <p>Hello {user.email}</p>
}
