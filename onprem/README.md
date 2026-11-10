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

Submit the isolated image to the existing Cloud Build/Artifact Registry project:

```sh
revision=$(git rev-parse HEAD)
image="us-east4-docker.pkg.dev/central-beach-194106/valuewholesale/valuewholesale-onprem-ga:${revision}"
gcloud builds submit onprem --project central-beach-194106 \
  --config onprem/cloudbuild.yaml \
  --substitutions="_IMAGE=${image},_REVISION=${revision}"
```

The upload excludes local credentials, environments, tests, and caches. The build checks the
packaged application import, Iris version, UI, and member fixture before publishing. Record
the resulting image digest and use that digest for the port-8083 cutover. Keep the existing
suite CA in a read-only runtime mount and point `SSL_CERT_FILE` to its combined public/private
trust bundle when the private Gateway endpoint uses the demo CA.

After the suite resources and scoped credentials pass their live checks, deploy the exact
image digest with the dedicated script:

```sh
./scripts/deploy_onprem_ga.sh \
  'us-east4-docker.pkg.dev/central-beach-194106/valuewholesale/valuewholesale-onprem-ga@sha256:ACTUAL_DIGEST' \
  /absolute/path/to/.env.onprem-ga /absolute/path/to/onprem-ca.crt
```

Use unquoted Docker `KEY=VALUE` lines in the environment file. The script checks a temporary
localhost-only canary before replacing port 8083. It retains the stopped previous container
and root-only configuration/metadata under `/var/backups/valuewholesale-onprem-ga/` on the VM.
If the replacement fails its readiness check, it restarts the previous container. It does not
create infrastructure, regenerate credentials, or alter the other demo containers.

Pass a fourth argument `canary` to stop after starting the new container on the VM's
localhost port 18083, leaving the public demo unchanged. Run the functional check against
that canary (or the deployed port 8083):

```sh
uv run --directory onprem python ../scripts/smoke_onprem_ga.py \
  --base-url http://VM_ADDRESS:8083 --output /private/path/ga-smoke.json
```

The check creates synthetic member sessions/conversations and one explicit shopping
preference. It verifies published FAQ bypass, live shopping inference and Gateway calls,
member memory separation, disconnect/cursor replay, retry deduplication, and retained results.
It does not claim a particular worker served either request. The localhost canary is reachable
only from the VM; run the check there or through an authenticated local tunnel. Remove only
`valuewholesale-onprem-ga-canary` after its checks, before running the final deploy.

For a live replay check across independent processes, start a second temporary container with
the same image, environment, CA bundle, and stable session secret, without publishing a host
port. Supply its independently inspected bridge address as `--replay-base-url`. The check
copies only the synthetic session cookie to that explicitly trusted peer in memory, resumes
the cursor there, and compares saved results from both endpoints. Retain the two container IDs
with the receipt as evidence, then remove the temporary peer. Individual Uvicorn worker
assignment remains unobserved.

Native Mesh blocks each Redis stream read for its heartbeat interval (15 seconds by default).
`REDIS_SOCKET_TIMEOUT=30` allows that blocking read to finish; an eight-second timeout prevents
the SDK stream builder from starting. The deployment preflight calls the installed SDK's
stream-settings validator before uploading any configuration.

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
[the presenter walkthrough](../docs/demo-valuewholesale-onprem.md) for repeatable checks and
content updates. The current deployment is licensed and provisioned; scoped keys remain in
the ignored runtime environment file.

On November 10, 2026, the licensed GKE suite and this application passed live canary checks
using application source commit `239d72fca2297d2c7fc47b720fd79c7777057487`. Its verified image is
`us-east4-docker.pkg.dev/central-beach-194106/valuewholesale/valuewholesale-onprem-ga@sha256:60fa2beaf97186b09eaf54215f0ab6d7a941a48b32bc02f74474ba5a43886efb`.
The pickup FAQ returned published version 1 with zero model calls. A live shopping turn used
six model calls plus member/product/inventory Gateway tools; an explicit preference used two
model calls and survived a new session while remaining absent for another member. Hybrid
RRF search, public relationship traversal, and handle operations passed separate Gateway checks;
the application shopping turn chose lexical search.

The native Mesh check disconnected during generation, reconnected to a distinct temporary
container/PID using the same Redis 8.6.2 catalog and shared session secret, resumed from the
last event ID, and verified retry deduplication, identical retained results from both containers,
and cross-member denial. The peer published no host port and was removed afterward. Operator
receipts retain the two container identities and trace IDs; exact Uvicorn worker assignment was
not observed. Final deployment to public port 8083 then passed the dedicated readiness gate.
Public health reports all seven services configured/ready, and browser verification displayed
the published returns FAQ version 1 and trace. Other demo containers are unchanged.
