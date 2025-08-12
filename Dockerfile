# Use a standard Python base image
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

# RUN useradd -m -u 1000 user
# USER user
# ENV PATH="/home/user/.local/bin:$PATH"


# # Set the working directory inside the container
# WORKDIR /app

# # Install system-level dependencies required for your Python packages
# # ghostscript is essential for camelot-py
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ghostscript \
#     && rm -rf /var/lib/apt/lists/*

# # Copy the requirements file first to leverage Docker layer caching
# COPY requirements.txt .

# # Install the Python packages
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy all your application files into the container
# COPY . .

# # Expose the port the app will run on (uvicorn's default is 8000)
# EXPOSE 8000

# # The command to run your FastAPI application when the container starts
# # The host "0.0.0.0" is important to make the app accessible from outside the container
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]