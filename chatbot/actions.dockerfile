# Extend the official Rasa SDK image
FROM rasa/rasa-sdk:3.6.2

USER root


# Use subdirectory as working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt



# Copy actions folder to working directory
COPY ./actions /app/actions
COPY entrypoint.sh /app/entrypoint.sh

# Set executable permission for entrypoint
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Don't run as root user
USER 1001


