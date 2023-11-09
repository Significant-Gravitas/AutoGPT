# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RSA cryptography signer and verifier."""


try:
    # Prefer cryptograph-based RSA implementation.
    from google.auth.crypt import _cryptography_rsa

    RSASigner = _cryptography_rsa.RSASigner
    RSAVerifier = _cryptography_rsa.RSAVerifier
except ImportError:  # pragma: NO COVER
    # Fallback to pure-python RSA implementation if cryptography is
    # unavailable.
    from google.auth.crypt import _python_rsa

    RSASigner = _python_rsa.RSASigner  # type: ignore
    RSAVerifier = _python_rsa.RSAVerifier  # type: ignore
