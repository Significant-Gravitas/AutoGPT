# 🔐 Security Policy

## Overview

We take the security of this project seriously and appreciate your help in keeping it safe for everyone.  
This document explains how to **responsibly disclose vulnerabilities**, what versions are **supported**, and how we **respond to reports**.

---

## 🧭 Scope

Security reports apply only to **actively maintained code**.  
> ⚠️ Code in the `classic/` folder or any directory explicitly marked as **deprecated** is **not supported** and will not receive fixes.

---

## 🪪 Reporting Security Issues

If you believe you’ve discovered a security vulnerability, **please report it privately**.  
Do **not** open public GitHub issues, discussions, or pull requests for security-related matters.

### 📨 Reporting Channels

- **Primary:** [GitHub Security Advisory](https://github.com/Significant-Gravitas/AutoGPT/security/advisories/new)  
- **Backup Email:** security@yourdomain.com *(replace with your team’s contact)*  
- *(Optional)* **Bounty Program:** [Huntr.dev](https://huntr.com/repos/significant-gravitas/autogpt)

---

## 🧩 Reporting Guidelines

When submitting a security report, please include:

1. A clear description of the vulnerability.  
2. Steps to reproduce the issue.  
3. Affected version, commit hash, or environment details.  
4. Potential impact (e.g., data leak, privilege escalation).  
5. Suggested mitigations (if any).

We welcome reports from anyone — researchers, users, or developers.

---

## ⏱️ Our Response Process

| Step | Action | Target Time |
|------|---------|-------------|
| 1️⃣ | Acknowledge report | Within **14 business days** |
| 2️⃣ | Validate and reproduce | Within **30 business days** |
| 3️⃣ | Develop and test fix | Within **60 business days** |
| 4️⃣ | Release patch | Within **90 business days** |
| 5️⃣ | Coordinate public disclosure | Within **30 days** after patch |

Total responsible disclosure window: **up to 120 days**.

---

## 🚫 Out of Scope

We will **not** accept or fix reports related to:

- Legacy code in `classic/` or any deprecated folder  
- Known issues documented in release notes  
- Vulnerabilities in third-party dependencies (should be reported upstream)  
- Social engineering, phishing, or denial-of-service (DoS) attacks  
- Misconfigurations in user environments or custom forks

---

## 🧰 Supported Versions

| Version | Supported | Notes |
|----------|------------|-------|
| Latest stable release (master branch) | ✅ | Receives full security updates |
| Development builds (`pre-master`) | ✅ | Reviewed for active testing |
| `classic/` folder (deprecated) | ❌ | Unsupported legacy code |
| Older releases | ❌ | No longer maintained |

---

## 🛡️ Security Best Practices for Users

When deploying or developing with this project:

1. Use the **latest stable release**.  
2. Keep all **dependencies updated** (`pip install --upgrade -r requirements.txt`).  
3. Run in **isolated environments** (e.g., Docker, venv).  
4. Store credentials and API keys securely — never commit them.  
5. Follow **principle of least privilege** in API and service configurations.  
6. Avoid using code from the `classic/` folder.  
7. Monitor for new advisories regularly.

---

## 🔒 Secure Development Practices

Our team follows the following security guidelines internally:

- Code reviewed for injection, deserialization, and sandbox vulnerabilities.  
- Continuous integration includes static analysis and dependency scanning.  
- Regular Python runtime updates (Python ≥ 3.13 recommended).  
- Dependencies locked using `requirements.txt` or `poetry.lock`.  
- Minimal use of system calls or unsafe file access.

---

## 🧬 Supply Chain & Dependency Security

We use automated tools to monitor dependency vulnerabilities:
- [Dependabot](https://github.com/dependabot)
- [PyUp.io](https://pyup.io)
- [Safety](https://pyup.io/safety/)

If a vulnerability is discovered in a dependency:
- We upgrade or patch within **14 days** of disclosure.
- Major version upgrades are tested before release.

---

## 💰 Security Bounties *(optional)*

We value the security community’s contributions.  
If your report qualifies under our bounty program (via Huntr.dev or similar), you may be eligible for a reward.  
Bounty amounts depend on **severity**, **impact**, and **quality of report**.

---

## 📬 Contact

- **Security Email:** security@yourdomain.com  
- **PGP Key:** [Download Here](https://yourdomain.com/pgp-key.asc)  
- **Preferred Languages:** English  

If you require encrypted communication, please contact us for our latest PGP key fingerprint.

---

## 🗓️ Change Log

| Date | Update |
|------|---------|
| Nov 2024 | Policy introduced |
| Oct 2025 | Expanded scope, added bounty & dependency sections |

---

_Last updated: October 2025_
