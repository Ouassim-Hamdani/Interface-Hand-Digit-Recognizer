FROM python:3.12.6

WORKDIR /app

# Copy project files
COPY models/ /app/models/
COPY data/ /app/data/
COPY src/ /app/src/
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN apt-get update

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get install -y libgl1-mesa-glx
# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "src/app.py"]