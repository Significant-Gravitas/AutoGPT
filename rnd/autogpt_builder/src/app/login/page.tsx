"use client";
import useUser from '@/hooks/useUser';
import { login, signup } from './actions'
import useSupabase from '@/hooks/useSupabase';
import { Button } from '@/components/ui/button';

export default function LoginPage() {
  const supabase = useSupabase();
  const { user, isLoading } = useUser();

  if (isLoading) {
    return <p>Loading...</p>
  }

  if (user) {
    return (
      <div>
        <p>Hello {user.email}</p>
        <Button onClick={() => supabase.auth.signOut()}>Sign out</Button>
      </div>
    )
  }

  async function handleSignInWithGoogle() {
    console.log('handleSignInWithGoogle');
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `http://localhost:3000/auth/callback`,
      },
    })
    console.log('data', data);
    console.log('error', error);
    
    // Get Google provider_refresh_token
    // const { data, error } = await supabase.auth.signInWithOAuth({
    //   provider: 'google',
    //   options: {
    //     queryParams: {
    //       access_type: 'offline',
    //       prompt: 'consent',
    //     },
    //   },
    // })
    
  }

  return (
    <div>
      <Button onClick={() => handleSignInWithGoogle()}>
        Sign in with Google
      </Button>
      <form>
        <label htmlFor="email">Email:</label>
        <input id="email" name="email" type="email" required />
        <label htmlFor="password">Password:</label>
        <input id="password" name="password" type="password" required />
        <button formAction={login}>Log in</button>
        <button formAction={signup}>Sign up</button>
      </form>
    </div >
  )
}
