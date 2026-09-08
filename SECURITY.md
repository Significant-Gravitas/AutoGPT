# Security Policy

## 1. Purpose

This Security Policy sets out the process for reporting security vulnerabilities in the AutoGPT platform, the standards expected of security researchers, and the approach taken by us in investigating, remediating, and disclosing vulnerabilities.

We are committed to maintaining the security and integrity of our systems and welcome responsible disclosure of vulnerabilities.

## 2. Reporting Security Vulnerabilities

If you believe you have identified a security vulnerability, you must report it confidentially and in accordance with this policy.

**Do not disclose vulnerabilities publicly** (including via GitHub issues, discussions, pull requests, or social media) prior to coordinated disclosure.

### 2.1 Reporting Channels

- **GitHub Security Advisory (preferred):**\
  <https://github.com/Significant-Gravitas/AutoGPT/security/advisories/new>
- **Email:** <security@agpt.co>

Reports submitted through other channels may not be monitored or may result in delays.

## 3. Information to Include

To enable effective triage and remediation, reports should include:

- A clear description of the vulnerability, including affected components
- The potential impact (e.g. confidentiality, integrity, availability)
- Step-by-step instructions to verify/exploit the vulnerability
- Version number, commit hash, or environment details
- Proof-of-concept code, scripts, or screenshots (where appropriate)
- Any known mitigations or suggested fixes

Incomplete reports may delay investigation.

## 4. Response and Remediation Process

We follow a coordinated vulnerability disclosure process.

### 4.1 Target Timelines

| Stage | Target |
| --- | --- |
| Initial acknowledgment | Within 5 business days |
| Initial assessment / triage | Within 14 business days |
| Remediation window | Up to 90 days |
| Public disclosure | Typically within 120 days of report |

These timelines are indicative and may be adjusted depending on:

- Severity and exploitability
- Complexity of remediation
- Risk to users or infrastructure

Where appropriate, we may accelerate disclosure for actively exploited vulnerabilities or extend timelines for complex fixes.

### 4.2 Communication

We will:

- Keep reporters informed of status where reasonably practicable
- Coordinate disclosure timing where possible
- Notify users through appropriate channels (e.g. advisories, release notes)

## 5. Coordinated Disclosure

You agree to:

- Refrain from public disclosure until:
  - A fix has been released; or
  - We have agreed on a disclosure timeline; or
  - 120 days have elapsed from initial report (whichever is earlier, unless otherwise agreed)

We reserve the right to disclose vulnerabilities earlier where necessary to protect users or systems.

## 6. Safe Harbor

We support good faith security research conducted in accordance with this policy.

We will not initiate legal action against you for:

- Accessing or testing systems within the defined scope
- Conducting research in good faith and in compliance with this policy

Provided that you:

- Comply with all applicable laws and regulations
- Do not exploit vulnerabilities for personal gain
- Do not access, modify, or exfiltrate data beyond what is strictly necessary
- Promptly report any discovered vulnerabilities

### 6.1 Limitations

This Safe Harbour:

- Does not apply to activities outside the scope of this policy
- Does not extend to third-party systems or services
- Does not override applicable laws or regulatory obligations
- Does not create any contractual relationship, employment relationship, or entitlement to compensation

Where a third party initiates legal action, we may, at our discretion, confirm that the research was conducted in accordance with this policy.

## 7. Testing Guidelines

All testing must be conducted responsibly and proportionately.

You must:

- **Minimise impact**: only perform actions necessary to demonstrate the vulnerability
- **Avoid data compromise:** do not access, download, or retain personal or confidential data unless strictly necessary and then only minimally
- **Preserve system integrity:** do not alter, corrupt, or destroy data
- **Protect availability**: do not conduct denial-of-service, stress, or load testing on production systems
- **Use appropriate environments**: use test or staging environments where available
- **Cease testing once validated**: stop testing once sufficient evidence is obtained

Any activity that risks harm to users, systems, or the confidentiality, integrity or availability of data is strictly prohibited.

Failure to comply with these guidelines may result in exclusion from Safe Harbor protections.

## 8. Scope

### 8.1 In Scope

- Latest production release on the master branch
- Active development code intended for future release

### 8.2 Out of Scope

- Code within the classic/ directory (unsupported)
- Legacy or unsupported versions
- Third-party dependencies (unless the issue arises directly from AutoGPT’s implementation)
- Infrastructure or systems not owned or controlled by us

We reserve the right to determine whether a reported issue falls within scope.

## 9. Recognition

We recognise the contribution of security researchers.

Subject to your consent:

- You will be credited in relevant security advisories; and
- You may be listed in our Security Acknowledgments section

We do not currently operate a paid bug bounty programme and no compensation is guaranteed.

## 10. User Security Responsibilities

Users of AutoGPT are responsible for maintaining secure deployments.

We recommend that users:

- Use the latest supported version
- Monitor and apply security updates promptly
- Review published security advisories
- Maintain secure configuration and access controls
- Avoid using default passwords and encryption keys
- Keep dependencies up to date
- Avoid use of deprecated components (including the classic/ folder)

## 11. Security Advisories

- GitHub Security Advisories:\
  <https://github.com/Significant-Gravitas/AutoGPT/security/advisories>
- Huntr disclosures:\
  <https://huntr.com/repos/significant-gravitas/autogpt>

## 12. Disclaimer

This policy:

- Does not grant permission to access systems outside its defined scope
- Does not create any legal obligation to provide compensation
- May be updated from time to time without prior notice

## 13. Acknowledgments

We thank the following security researchers for contributing responsibly to improving the security of AutoGPT:

- [@lukas-eu](https://github.com/lukas-eu) – 3 advisories
- [@AgentSec](https://github.com/AgentSec) – 11 advisories
- Joshua Rogers ([@MegaManSec](https://github.com/MegaManSec)) – 2 advisories
- Gecko Security ([@geckosecurity](https://github.com/geckosecurity)) – 1 advisory
- JJ ([@jjjutla](https://github.com/jjjutla)) – 1 advisory
- Artemiy ([@nkoorty](https://github.com/nkoorty)) – 1 advisory
- [@rahulgovind](https://github.com/rahulgovind) – 1 advisory
- Panuganti Siva Aditya ([@sivaadityacoder](https://github.com/sivaadityacoder)) – 1 advisory
- Sunwoo Lee ([@programsurf](https://github.com/programsurf)) – 2 advisories
- [@daeungdaeung](https://github.com/daeungdaeung) – 2 advisories
- Seunghyun Yoon ([@yoonsh](https://github.com/yoonsh)) – 1 advisory
- @lubroai – 2 advisories
- [@ltduc147](https://github.com/ltduc147) – 1 advisory
- [@222n5](https://github.com/222n5) – 3 advisories
- [@ygboy777-alt](https://github.com/ygboy777-alt) – 1 advisory
- [@S4nso](https://github.com/S4nso) – 1 advisory
- [@CYC4D](https://github.com/CYC4D) – 1 advisory
- [@Mirr2](https://github.com/Mirr2) – 1 advisory
- Minjoon Gregorio Kim ([@Greg-Kim](https://github.com/Greg-Kim)) – 1 advisory
- Dirstibone ([@1024wlsdud](https://github.com/1024wlsdud)) – 1 advisory
- [@johnatzeropath](https://github.com/johnatzeropath) – 1 advisory
- [@LeftenantZero](https://github.com/LeftenantZero) – 1 advisory
- [@dxlerYT](https://github.com/dxlerYT) – 1 advisory
- Jiaqi Luo ([@lhahah](https://github.com/lhahah)) – 1 advisory
- Pavan Nallamothu ([@pavanchow](https://github.com/pavanchow)) – 1 advisory
- Johnathan ([@TrebledJ](https://github.com/TrebledJ)) – 1 advisory
- [@sai-sh](https://github.com/sai-sh) – 1 advisory
- [@Acce1erator-R](https://github.com/Acce1erator-R) – 1 advisory
- Edward-x ([@YLChen-007](https://github.com/YLChen-007)) – 1 advisory
- 狗and猫 ([@Fushuling](https://github.com/Fushuling)) – 1 advisory
- RacerZ ([@RacerZ-fighting](https://github.com/RacerZ-fighting)) – 1 advisory
- Ace ([@Fried-Squid](https://github.com/Fried-Squid)) – 1 advisory
- David Carliez ([@DavidCarliez](https://github.com/DavidCarliez)) – 1 advisory
- [@long2809-exploi](https://github.com/long2809-exploi) – 1 advisory

**Last updated: September 2026**
