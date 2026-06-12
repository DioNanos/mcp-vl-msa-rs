# Security Policy

## Supported versions

Only the latest released line receives security fixes.

| Version | Supported |
| ------- | --------- |
| 0.4.x   | Yes       |
| < 0.4   | No        |

## Reporting a vulnerability

Please report security issues **privately**. Do not open a public issue for
anything you believe is a vulnerability.

Preferred channel: GitHub Security Advisories — use the **"Report a
vulnerability"** button on this repository's *Security* tab. This keeps the
report private until a fix is available.

Fallback: email **dev@mmmbuto.com**.

When reporting, include the affected version, your platform/target, and the
steps to reproduce.

### Response times

This is a single-maintainer project, so handling is **best-effort**. Expect an
initial acknowledgement within a few days and a fix on a timeline that depends
on severity and available time. Please be patient.

## Scope notes

The server is **stdio, local-first**. It speaks to an MCP client over standard
input/output on the same machine and does not open a network listener.
Collections are stored on the **user's local disk**. Data security therefore
follows local filesystem permissions and the trust boundary of the user account
running the server.
