# Data Retention and Privacy Policy

## Purpose

This policy defines how long each class of data is kept, how it is deleted, and
who may authorise an exception. It complements the IT Security Policy, which
defines the classification tiers referenced below.

## Retention periods

| Data class | Retention period | Notes |
| ---------- | ---------------- | ----- |
| Authentication and audit logs | 400 days | Matches the logging requirement in the security policy |
| Application debug logs | 30 days | Must not contain Restricted data |
| Customer transaction records | 7 years | Statutory requirement; archived, not deleted |
| Support tickets | 3 years after closure | |
| Backup snapshots | 35 days | Rolling; older snapshots are purged automatically |
| Marketing analytics events | 13 months | Aggregated after 90 days |

Data past its retention period must be deleted, not merely hidden. Deletion
runs as a scheduled job and writes an entry to the audit log naming the job, the
data class, and the number of records removed. The job never logs the record
contents.

## Personal data

Personal data means any information relating to an identifiable person,
including names, account numbers, national identifiers, and device identifiers.
Two principles apply:

1. **Minimisation.** Collect personal data only when a specific, documented
   purpose requires it. If a field has no owner and no purpose, it is deleted.
2. **Purpose limitation.** Personal data collected for one purpose may not be
   repurposed without a new lawful basis.

Requests from individuals to see or delete their personal data are handled by
the Data Protection Officer and must be answered within 30 calendar days.

## Right to erasure

When an erasure request is approved, records in live systems are deleted within
7 days. Records held in backups are not individually deleted; instead they are
left to expire naturally within the 35-day backup window, and the erasure is
recorded so that a restored backup can be re-processed.

## Cross-border transfer

Restricted data may not leave the primary region without an approved transfer
assessment. Where a third-party processor is involved, a data-processing
agreement must be in place before any data is sent, and the vendor must appear on
the approved vendor register.

## Exceptions

Any deviation from these periods requires a written exception approved by the
Data Protection Officer and recorded in the exception register. Exceptions are
reviewed every 6 months and expire after 12 months unless renewed.
