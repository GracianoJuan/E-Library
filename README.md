# E-Library Docker Deployment Guide

This repository is set up for a containerized deployment with four services:

- PostgreSQL 16 for persistence
- Python 3.10.6 backend for the API and migrations
- Next.js frontend on Node 22.17.0
- Nginx as the reverse proxy and traffic entrypoint

The Docker entrypoint for the backend waits for PostgreSQL to become healthy, then runs `python migration_runner.py upgrade` followed by `python migrations/migrate_csv.py` before starting the API server.

## Development Setup Guide

### 1. Prepare local environment variables

From the repository root, create the development env file from the example:

```powershell
Copy-Item .env.dev.example .env.dev
```

Review the values and adjust them if needed. The default development layout expects:

- `POSTGRES_DB=elibrary_db`
- `POSTGRES_USER=postgres`
- `POSTGRES_PASSWORD=postgres`
- `POSTGRES_PORT=5432`
- `BACKEND_PORT=8000`
- `FRONTEND_PORT=3000`
- `NGINX_PORT=8080`
- `CLENT_URL=http://localhost:8080`
- `JWT_SECRET=...`
- `DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/elibrary_db`
- `NEXT_PUBLIC_API_URL=http://localhost:8080/api`

### 2. Build and start the development stack

Run the following from the repository root:

```powershell
docker compose -f docker-compose.dev.yml up -d --build
```

This starts the full local stack with hot reloading:

- Backend source is bind-mounted into the container.
- Frontend source is bind-mounted into the container.
- PostgreSQL is exposed on the host for debugging.
- Nginx routes all browser traffic through a single local entrypoint.

### 3. Verify migrations and CSV import executed

The backend container performs both steps automatically after PostgreSQL passes its health check. Verify the run with:

```powershell
docker compose -f docker-compose.dev.yml logs -f backend
```

Look for these messages:

- `Database is ready`
- `Upgrading to head...`
- `✓ Upgrade completed successfully!`
- `Successfully migrated ... books from CSV to database`

If you want to inspect the database directly, use the exposed Postgres port and connect with the credentials from `.env.dev`.

### 4. Access the app locally through the proxy

Open the proxy entrypoint in your browser:

- Frontend: `http://localhost:8080`
- API: `http://localhost:8080/api`

The Nginx development proxy forwards `/api/*` to the backend and strips the `/api` prefix before forwarding.

### 5. Stop the development stack

```powershell
docker compose -f docker-compose.dev.yml down
```

To remove the database volume as well:

```powershell
docker compose -f docker-compose.dev.yml down -v
```

## Production Deployment Guide

### Architecture overview

The production stack differs from development in three important ways:

- Isolation: the app services communicate over a dedicated private Docker network.
- Security: the database is not published to the host, and secrets come from a production env file.
- Optimization: the frontend uses a multi-stage standalone build, and the backend starts only after PostgreSQL reports healthy.

### 1. Configure production environment variables safely

Create the production env file from the example and store it securely:

```powershell
Copy-Item .env.prod.example .env.prod
```

Use strong, unique values for:

- `POSTGRES_PASSWORD`
- `JWT_SECRET`
- `DATABASE_URL`
- `CLENT_URL`
- `NEXT_PUBLIC_API_URL`

Do not commit `.env.prod` to source control. Keep it on the deployment host or inject it through your secret-management workflow.

### 2. Prepare SSL files

Production Nginx expects certificates in `./ssl`:

- `./ssl/fullchain.pem`
- `./ssl/privkey.pem`

Place your issued certificates there before starting the production stack.

### 3. Build and deploy in detached mode

Run the production environment with:

```powershell
docker compose -f docker-compose.prod.yml up -d --build
```

This command:

- Builds the optimized frontend production image.
- Starts PostgreSQL without host port exposure.
- Runs the backend migration-and-import bootstrap sequence after database health checks pass.
- Publishes only Nginx on ports `80` and `443`.

### 4. Check production health and logs

Inspect logs for a specific service:

```powershell
docker compose -f docker-compose.prod.yml logs -f nginx
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f db
```

Check container status:

```powershell
docker compose -f docker-compose.prod.yml ps
```

Useful production checks:

- Backend startup should show the database wait, migration, and CSV import steps.
- Nginx should return a `301` from port `80` to `https` and proxy traffic on `443`.
- PostgreSQL should remain healthy with no exposed host port.

### 5. Production troubleshooting

If startup fails, check these first:

1. Database health:

	```powershell
	docker compose -f docker-compose.prod.yml logs -f db
	```

2. Backend migration output:

	```powershell
	docker compose -f docker-compose.prod.yml logs -f backend
	```

3. Nginx certificate mounting:

	```powershell
	docker compose -f docker-compose.prod.yml logs -f nginx
	```

4. Service state:

	```powershell
	docker compose -f docker-compose.prod.yml ps
	```

If the backend cannot connect to the database, confirm that `DATABASE_URL` points at the `db` service hostname and that the Postgres password matches the production env file.


