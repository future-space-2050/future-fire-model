FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Set up cache directories
ENV TRANSFORMERS_CACHE=/tmp/cache/
ENV HUGGINGFACE_HUB_CACHE=/tmp/cache/
RUN mkdir -p /tmp/cache && chmod 777 /tmp/cache

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /tmp/cache
WORKDIR /app
RUN chown appuser:appuser /app

# Copy requirements first for caching
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY --chown=appuser:appuser who_to_follow/DataSet/ /app/who_to_follow/DataSet/
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port for Flask
EXPOSE 5000

# CMD ["gunicorn","python", "main.py"] before
CMD ["gunicorn", "-b", "0.0.0.0:5000", "main:app"]