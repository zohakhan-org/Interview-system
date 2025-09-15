#!/bin/bash

# Wait for database to be ready (if using PostgreSQL)
# while ! nc -z $DB_HOST $DB_PORT; do
#   echo "Waiting for database..."
#   sleep 1
# done

# Run database migrations (if needed)
# python -m app.db.migrate

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4