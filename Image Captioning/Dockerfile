# Use an official Python runtime as the parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available for the app to be accessed outside the container
EXPOSE 5000

# Define an environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Run the command to start your app when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
