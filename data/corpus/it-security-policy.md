# IT Security Policy

## Scope

This policy applies to all production systems, engineering workstations, and
third-party integrations operated by the Platform Engineering group. It is
reviewed annually by the Security Office.

## Authentication

All interactive logins to production consoles must use multi-factor
authentication (MFA). Password-based single-factor login is prohibited for
administrative accounts. Service accounts must not use interactive login at all;
they authenticate with short-lived tokens issued by the internal identity
service.

Passwords for any remaining interactive account must be rotated every 90 days.
The minimum length is 14 characters, and the previous five passwords may not be
reused. Where a system supports it, passkeys are preferred over passwords.

## Access reviews

Privileged access is reviewed every quarter. The reviewer is the system owner,
not the requester. Any access not re-confirmed within 10 business days is
revoked automatically by the access-management job.

## Data classification

Data is classified into three tiers:

- **Public** — may be shared externally without restriction.
- **Internal** — default for engineering artefacts; sharing outside the company
  requires an approved exception.
- **Restricted** — includes customer PII, account numbers, and authentication
  secrets. Restricted data may never be copied to laptops, personal cloud
  storage, or unmanaged databases. Moving restricted data requires a documented
  data-transfer request approved by the Data Protection Officer.

## Incident response

Security incidents are triaged by severity:

| Severity | Definition | First response target |
| -------- | ---------- | --------------------- |
| SEV-1 | Confirmed breach or data exfiltration | 15 minutes |
| SEV-2 | Credible attempt or exposure without confirmed loss | 1 hour |
| SEV-3 | Policy violation or suspicious activity | 1 business day |

For a suspected SEV-1, the responder must page the on-call security engineer
immediately, preserve logs before any remediation, and open an incident channel.
The incident commander is responsible for all external communication; engineers
must not contact press or customers directly.

## Logging

Authentication events, privilege changes, and access to Restricted data are
logged and retained for 400 days. Application debug logs must never contain
passwords, tokens, or full account numbers.
