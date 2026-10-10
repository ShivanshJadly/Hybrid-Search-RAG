# Incident Response Runbook

## Severity Levels

Incidents are classified from SEV1 to SEV4. The severity drives who gets paged
and how quickly updates are expected, so classify early and adjust rather than
waiting for certainty.

SEV1 means a complete outage of a customer-facing service, or any confirmed data
loss. It pages the on-call engineer, the on-call manager and the incident
commander rotation simultaneously. Updates are expected every fifteen minutes.

SEV2 means significant degradation with a workaround available. It pages the
on-call engineer only. Updates every thirty minutes.

SEV3 covers issues affecting a small subset of users with no customer-visible
outage. It creates a ticket rather than paging. SEV4 is for internal-only
problems that need tracking but no urgency.

## First Fifteen Minutes

Do not start debugging before declaring. Open an incident channel, state the
severity, and name an incident commander. The commander does not debug; their
job is coordination, communication and deciding when to escalate.

Check the deployment ledger first. A substantial majority of incidents follow a
deploy within the previous hour, and rolling back is faster than diagnosing.
If a deploy went out recently and the timing is suspicious, roll back before
investigating further. Diagnosis is much easier once customers are unaffected.

## Common Failure Modes

### Database connection exhaustion

Symptoms are rising p99 latency across every endpoint at once, followed by
`ERR_DB_5031` in application logs. The connection pool is configured through
`DATABASE_POOL_SIZE` and defaults to twenty per instance.

The usual trigger is an autoscaling event that raises instance count past what
the database can accept. Total connections equal pool size multiplied by
instance count, and it is easy to scale the application tier past the database
limit without noticing.

Mitigation is to cap instance count temporarily, then either raise the database
connection limit or lower the per-instance pool size. Lowering the pool size is
usually correct; most services are configured with a pool far larger than their
actual concurrency requires.

### Cache stampede

Symptoms are a sudden spike in database load with no corresponding increase in
request volume. This happens when a popular cache key expires and every request
that would have hit the cache goes to the database simultaneously.

The fix in the moment is to warm the key manually. The permanent fix is jittered
expiry, which is available in the shared cache client by setting
`CACHE_TTL_JITTER_PERCENT` to something above zero. Ten percent is a reasonable
starting point and largely eliminates synchronised expiry.

### Queue backlog

Symptoms are growing consumer lag with normal producer rates. Check consumer
health before assuming a throughput problem, because the most common cause is a
consumer crash-looping on a single poisoned message rather than genuine
overload.

A poisoned message is identifiable by a repeating error at a fixed offset in the
consumer logs. The mitigation is to move that message to the dead letter queue
and let the consumer proceed. The `QUEUE_MAX_REDELIVERIES` setting controls how
many attempts happen before automatic dead-lettering; it defaults to five, which
is often too high for consumers that fail slowly.

## Communication

Customer-facing updates go out through the status page and are written by the
incident commander, not the engineers debugging. Internal updates go in the
incident channel.

Say what is known, what is not known, and when the next update will come. Do not
speculate about cause in customer-facing communication. An early wrong cause
reported publicly is far more damaging than saying an investigation is ongoing.

## After the Incident

Every SEV1 and SEV2 requires a written retrospective within five working days.
The retrospective is blameless and focuses on the conditions that allowed the
failure, not on who acted.

The output that matters is action items with owners and dates. A retrospective
that produces a narrative and no actions has not achieved anything. Action items
are tracked to completion and reviewed monthly.
