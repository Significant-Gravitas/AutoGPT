"use client";
import useUser from '@/hooks/useUser';
import { login, signup } from './actions'
import { Button } from '@/components/ui/button';
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { useForm } from 'react-hook-form';
import { Input } from '@/components/ui/input';
import { z } from "zod"
import { zodResolver } from "@hookform/resolvers/zod"
import { PasswordInput } from '@/components/PasswordInput';
import { FaGoogle, FaGithub, FaDiscord, FaSpinner } from "react-icons/fa";
import useSupabase from '@/hooks/useSupabase';

const loginFormSchema = z.object({
  email: z.string().email().min(2).max(64),
  password: z.string().min(6).max(64),
})

export default function LoginPage() {
  const supabase = useSupabase();
  const { user, isLoading } = useUser();

  const form = useForm<z.infer<typeof loginFormSchema>>({
    resolver: zodResolver(loginFormSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  })

  if (!supabase) {
    return <div>User accounts are disabled because Supabase client is unavailable</div>
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
    const { data, error } = await supabase!.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `http://localhost:3000/auth/callback`,
      },
    })

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

  const onLogin = async (data: z.infer<typeof loginFormSchema>) => {
    login(data)
  }

  const onSignup = async (data: z.infer<typeof loginFormSchema>) => {
    if (await form.trigger()) {
      signup(data)
    }
  }

  return (
    <div className="flex items-center justify-center h-[80vh]">
      <div className="w-full max-w-md p-8 bg-white rounded-lg shadow-md space-y-6">
        <div className='mb-6 space-y-2'>
          <Button className="w-full" onClick={() => handleSignInWithGoogle()} variant="outline" type="button" disabled={isLoading}>
            {isLoading ? (
              <FaSpinner className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <FaGoogle className="mr-2 h-4 w-4" />
            )}{" "}
            Sign in with Google
          </Button>
          {/* <Button className="w-full" onClick={() => handleSignInWithGoogle()} variant="outline" type="button" disabled={isLoading}>
            {isLoading ? (
              <FaSpinner className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <FaGithub className="mr-2 h-4 w-4" />
            )}{" "}
            Sign in with GitHub
          </Button>
          <Button className="w-full" onClick={() => handleSignInWithGoogle()} variant="outline" type="button" disabled={isLoading}>
            {isLoading ? (
              <FaSpinner className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <FaDiscord className="mr-2 h-4 w-4" />
            )}{" "}
            Sign in with Discord
          </Button> */}
        </div>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onLogin)}>
            <FormField
              control={form.control}
              name="email"
              render={({ field }) => (
                <FormItem className='mb-4'>
                  <FormLabel>Email</FormLabel>
                  <FormControl>
                    <Input placeholder="user@email.com" {...field} />
                  </FormControl>
                  <FormDescription>
                    This is your email.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Password</FormLabel>
                  <FormControl>
                    <PasswordInput placeholder="password" {...field} />
                  </FormControl>
                  <FormDescription>
                    This is your password.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className='flex w-full space-x-4 mt-6 mb-6'>
              <Button className='w-1/2 flex justify-center' type="submit">Log in</Button>
              <Button
                className='w-1/2 flex justify-center'
                variant='secondary'
                type="button"
                onClick={form.handleSubmit(onSignup)}
              >
                Sign up
              </Button>
            </div>
          </form>
          <span className='text-secondary text-center text-sm'>
            By continuing you agree to everything
          </span>
        </Form>
      </div>
    </div>
  )
}
