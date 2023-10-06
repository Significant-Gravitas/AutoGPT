import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final GoogleSignIn googleSignIn = GoogleSignIn(
      clientId:
          "387936576242-iejdacrjljds7hf99q0p6eqna8rju3sb.apps.googleusercontent.com");

// Sign in with Google using redirect
// Sign in with Google using redirect
  Future<UserCredential?> signInWithGoogle() async {
    try {
      final GoogleSignInAccount? googleSignInAccount =
          await googleSignIn.signIn();
      if (googleSignInAccount != null) {
        final GoogleSignInAuthentication googleSignInAuthentication =
            await googleSignInAccount.authentication;
        final AuthCredential credential = GoogleAuthProvider.credential(
          accessToken: googleSignInAuthentication.accessToken,
          idToken: googleSignInAuthentication.idToken,
        );
        return await _auth.signInWithCredential(credential);
      }
    } catch (e) {
      print("Error during Google Sign-In: $e");
      return null;
    }
  }

// Sign in with GitHub using redirect
  Future<UserCredential?> signInWithGitHub() async {
    try {
      final GithubAuthProvider provider = GithubAuthProvider();
      return await _auth.signInWithPopup(provider);
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
