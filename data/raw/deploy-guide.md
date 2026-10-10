# Deployment Guide

This guide covers how services in the platform group are built, promoted and
rolled back. It applies to every service that lives in the `platform/` monorepo
and is deployed through the shared pipeline.

## Prerequisites

Before deploying anything you need access to the deployment role in the target
account, and your local environment must have the `DEPLOY_TOKEN` variable set.
The token is issued by the platform team and rotates every ninety days. If your
token has expired the pipeline fails immediately with `ERR_AUTH_4011` rather
than partway through, which is deliberate.

You also need the CLI installed. Version 2.4 or later is required because
earlier versions do not understand the new manifest format. Running an older
CLI against a current manifest produces a confusing schema error rather than a
clear version complaint, so check the version first when something looks wrong.

## Build Stage

Every commit to a release branch triggers a build. The build produces a
container image tagged with the short commit SHA, pushes it to the internal
registry, and records the manifest digest in the deployment ledger.

Builds are reproducible. Given the same commit, the same base image and the
same lockfile, the resulting digest is identical. If two builds of the same
commit produce different digests, something has leaked into the build context,
and the usual culprit is an untracked file in the working directory.

The build stage has a hard timeout of twenty minutes. Services that legitimately
need longer must set `BUILD_TIMEOUT_MINUTES` in their service manifest, and the
platform team reviews any value above forty.

## Promotion

Images move through three environments in order: development, staging and
production. Promotion is always forward and always by digest, never by tag.
Promoting by tag is what allows a different image to reach production than the
one that was tested, so the pipeline refuses tag-based promotion outright.

A promotion to staging happens automatically once the build succeeds and the
integration suite passes. Promotion to production requires an approval from
someone other than the commit author. The approval is recorded against the
digest, so re-promoting the same digest later does not require fresh approval.

### Canary Releases

Production promotion defaults to a canary rollout. Ten percent of traffic moves
to the new digest for fifteen minutes while the error rate and latency are
compared against the previous digest. If either metric degrades beyond the
configured threshold, the rollout halts automatically and the canary is drained.

The comparison thresholds live in `canary-policy.yaml` at the repository root.
The defaults are a 0.5 percent absolute increase in error rate or a 20 percent
relative increase in p99 latency. Services with naturally spiky traffic often
need looser thresholds, and setting them too tight causes rollouts to abort on
noise rather than on real regressions.

## Rollbacks

A rollback is a promotion to a previously known-good digest. It uses exactly the
same machinery as a forward deployment, which means rollbacks are tested by
every normal deploy and are not a separate untested path.

To roll back, find the last good digest in the deployment ledger and promote it.
The ledger keeps ninety days of history per service. Anything older has been
pruned and the image may no longer exist in the registry, so a rollback beyond
ninety days generally requires rebuilding from the tag.

Rollbacks skip the canary stage by default. The reasoning is that a digest which
previously served production traffic successfully does not need to re-earn
confidence, and the situation prompting a rollback is usually urgent. This
default can be overridden per service with `ROLLBACK_CANARY_ENABLED`.

### What a rollback does not do

A rollback reverts the running code. It does not revert database migrations,
feature flag state, or anything written to durable storage since the bad deploy
went out. This is the single most common misunderstanding about the rollback
process and the cause of most post-rollback incidents.

If the bad deploy ran a migration, rolling back the code without addressing the
migration usually leaves the older code reading a schema it does not understand.
Forward-only migrations that are safe against both the old and new code are the
standard mitigation, and services are expected to write migrations that way.

## Troubleshooting

If a deploy hangs in the promotion stage with no log output, the usual cause is
a missing health check endpoint. The pipeline waits for `/healthz` to return 200
and will wait until the stage timeout before failing with `ERR_DEPLOY_5030`.

If the pipeline reports `ERR_REGISTRY_4041`, the digest referenced in the
promotion no longer exists in the registry. This happens when promoting a digest
older than the registry retention window, which is currently sixty days for
development images and one year for anything that reached production.
