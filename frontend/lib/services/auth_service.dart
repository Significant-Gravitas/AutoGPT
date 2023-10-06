import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final GoogleSignIn googleSignIn = GoogleSignIn(
      clientId:
          "387936576242-iejdacrjljds7hf99q0p6eqna8rju3sb.apps.googleusercontent.com");

// Sign in with Google using popup
  Future<UserCredential?> signInWithGoogle() async {
    try {
      final GoogleAuthProvider googleProvider = GoogleAuthProvider();

      // Step 1: Detect the current hostname
      String hostname = Uri.base.host;

      // Step 2: Determine the redirect URI
      String redirectUri;
      if (hostname.contains('github.dev')) {
        // If running in Github Codespaces
        redirectUri = Uri.base.toString();
      } else {
        // For local development or other environments, set accordingly
        redirectUri = 'http://localhost:8000'; // Example for local development
      }

      // Step 3: Update OAuth 2.0 provider configuration dynamically
      googleProvider.setCustomParameters({'redirect_uri': redirectUri});

      // Use signInWithPopup instead of signInWithRedirect
      final result = await _auth.signInWithPopup(googleProvider);
      print(result);
      return result;
    } catch (e) {
      print("Error during Google Sign-In: $e");
      return null;
    }
  }

// Sign in with GitHub using popup
  Future<UserCredential?> signInWithGitHub() async {
    try {
      final GithubAuthProvider githubProvider = GithubAuthProvider();

      // Step 1: Detect the current hostname
      String hostname = Uri.base.host;

      // Step 2: Determine the redirect URI
      String redirectUri;
      if (hostname.contains('github.dev')) {
        // If running in Github Codespaces
        redirectUri = Uri.base.toString();
      } else {
        // For local development or other environments, set accordingly
        redirectUri = 'http://localhost:8000'; // Example for local development
      }

      // Step 3: Update OAuth 2.0 provider configuration dynamically
      githubProvider.setCustomParameters({'redirect_uri': redirectUri});

      // Use signInWithPopup instead of signInWithRedirect
      final result = await _auth.signInWithPopup(githubProvider);
      return result;
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
