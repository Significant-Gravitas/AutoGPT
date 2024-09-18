import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/services/auth_service.dart';

class FirebaseAuthView extends StatelessWidget {
  // TODO: This should be initialized in the main.dart instead of here
  final AuthService _authService = AuthService();

  FirebaseAuthView({super.key}); // Initialize the auth service

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            OutlinedButton(
              onPressed: () async {
                final user = await _authService.signInWithGoogle();
                if (user != null) {
                  print(
                      "Successfully signed in with Google: ${user.user?.displayName}");
                }
              },
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.blue,
                side: const BorderSide(color: Colors.blue, width: 2),
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Image.asset('assets/images/google_logo.svg.png', width: 24),
                  const SizedBox(width: 8),
                  const Text('Sign in with Google',
                      style: TextStyle(fontWeight: FontWeight.w300)),
                ],
              ),
            ),
            const SizedBox(height: 20),
            OutlinedButton(
              onPressed: () async {
                final user = await _authService.signInWithGitHub();
                if (user != null) {
                  print(
                      "Successfully signed in with GitHub: ${user.user?.displayName}");
                }
              },
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.black,
                side: const BorderSide(color: Colors.black, width: 2),
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Image.asset('assets/images/github_logo.svg.png', width: 24),
                  const SizedBox(width: 8),
                  const Text('Sign in with GitHub',
                      style: TextStyle(fontWeight: FontWeight.w300)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
