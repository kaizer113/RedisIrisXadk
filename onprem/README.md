# Value Wholesale on Redis Iris GA

This project owns only the Value Wholesale self-managed demo on port 8083. It uses a bounded
Google GenAI tool loop on Vertex, with published Playbooks, Intent Router, live Context Gateway
tools, scoped Agent Memory, Lens tracing, and Agent Mesh SSE. It has no preview API fallback.
The original app, its `pyproject.toml`, dependency lock, and Dockerfile remain separate.

```sh
uv sync --directory onprem --frozen
cp .env.onprem-ga.example .env.onprem-ga
# Fill the actual GA service resource IDs and scoped keys, then:
make dev-onprem
```

Runtime keys belong only in the ignored `.env.onprem-ga`. Keep operator tokens and suite
licenses out of this file and image. `ONPREM_SESSION_SECRET` must be stable across both workers
and redeployments; generate at least 32 random characters once. The member picker selects
fictional personas and issues signed session cookies; it is not a customer sign-in service.

Build only this application:

```sh
docker build -t valuewholesale-onprem-ga:1.0 onprem
docker run --rm --env-file .env.onprem-ga -p 8083:8080 \
  --label owner=lionel_giavelli --label skip_deletion=yes valuewholesale-onprem-ga:1.0
```

For Cloud Build, use `onprem/` as the build context and its `Dockerfile`; use a separate image
name such as `valuewholesale-onprem-ga`, and replace only container `valuewholesale-onprem-agent`
on `valuewholesale-demo`. Do not use the default app image or `scripts/deploy_vm.sh` for this
runtime. The latter also changes shared infrastructure and filters out the new GA settings.

Verify locally:

```sh
uv run --directory onprem ruff check .
uv run --directory onprem pytest -q
make lint
make test
```

The native Mesh test starts a temporary Redis bound to localhost with persistence disabled,
plus two HTTP application replicas. It tests disconnect, cursor replay on the other replica,
idempotent retry, retained-result recovery, and denial from another member session. It requires
a local `redis-server` version 8.6.2 or later; otherwise that test is skipped. It does not qualify
Redis Cloud, GKE, failover, or actual Iris service connectivity.

See [deployment and operator setup](../docs/onprem-iris.md) and
[the presenter walkthrough](../docs/demo-valuewholesale-onprem.md). The checked-in code and
content are ready for review; live GA service checks require the suite chart, license,
provisioned resources, Gateway data model, and scoped runtime keys.
