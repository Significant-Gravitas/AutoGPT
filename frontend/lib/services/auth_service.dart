import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final GoogleSignIn googleSignIn = GoogleSignIn(
      clientId:
          "387936576242-iejdacrjljds7hf99q0p6eqna8rju3sb.apps.googleusercontent.com");

// Sign in with Google using redirect
  Future<UserCredential?> signInWithGoogle() async {
    try {
      final GoogleAuthProvider googleProvider = GoogleAuthProvider();
      await _auth.signInWithRedirect(googleProvider);
      return await _auth.getRedirectResult();
    } catch (e) {
      print("Error during Google Sign-In: $e");
      return null;
    }
  }

// Sign in with GitHub using redirect
  Future<UserCredential?> signInWithGitHub() async {
    try {
      final GithubAuthProvider githubProvider = GithubAuthProvider();
      await _auth.signInWithRedirect(githubProvider);
      return await _auth.getRedirectResult();
    } catch (e) {
      print("Error during GitHub Sign-In: $e");
      return null;
    }
  }

  // Sign out
  Future<void> signOut() async {
    await _auth.signOut();
  }

  // Get current user
  User? getCurrentUser() {
    return _auth.currentUser;
  }
}
