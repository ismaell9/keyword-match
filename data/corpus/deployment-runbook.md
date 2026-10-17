# Deployment Runbook

## Pipeline overview

Every service builds through the same pipeline: unit tests, static analysis,
container image build, and a security scan. A change may only reach production
from a green pipeline on the `main` branch. Direct pushes to `main` are blocked;
all changes arrive through a pull request with at least one approval.

## Environments

There are three environments:

- **dev** — deploys automatically on every merge to `main`.
- **staging** — promotes automatically after dev has been healthy for 30 minutes.
- **production** — requires a manual approval in the deployment tool.

Staging must contain a schema-identical copy of the production database structure
(no production data). Deployments are blue/green: the new version receives no
traffic until its health check passes.

## Rollback

Every deployment keeps the previous image for 14 days. A rollback is a redeploy
of that image, not a rebuild, so it completes in under two minutes. Rollbacks do
not reverse database migrations. If a release included a destructive migration,
the rollback procedure is: disable the new code path with a feature flag, then
restore the database from the most recent snapshot.

## Service level objectives

Each production service defines:

- Availability SLO: 99.9% of requests succeed over a rolling 30 days.
- Latency SLO: 95th percentile response time under 400 milliseconds.
- Error budget: the shortfall between the SLO and perfect reliability. When more
  than 50% of the error budget for a service is consumed within 7 days, feature
  work for that service pauses until reliability work lands.

## On-call

Primary on-call rotates weekly. The responder acknowledges a page within 5
minutes. If not acknowledged, the page escalates to the secondary after 10
minutes and to the engineering manager after 20 minutes. A handover note is
required at every rotation change, listing open incidents and any temporary
mitigations still in place.

## Change freeze

A company-wide change freeze applies during the year-end financial close
(typically the last five business days of the fiscal year). Only SEV-1 fixes are
deployed during a freeze, and each requires director approval.
