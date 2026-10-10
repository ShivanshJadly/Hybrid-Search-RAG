# On-Call Quick Reference

A condensed version of the incident runbook, kept for the on-call handover.
Sections below are copied from the full runbook with light edits, which is
exactly the kind of duplication near-duplicate detection is meant to catch.

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

## Paging Contacts

The on-call rotation is published in the scheduling tool and mirrored to the
team channel every Monday morning. If nobody acknowledges a page within ten
minutes it escalates automatically to the on-call manager, and after a further
ten minutes to the engineering director.

Do not page an individual directly outside the rotation. Someone who is not on
call may be asleep, on leave, or simply unable to help, and paging around the
rotation delays the response rather than speeding it up.
