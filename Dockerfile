FROM python:3.6-slim
WORKDIR /srv/app
RUN apt update && apt upgrade -y && apt autoremove -y
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt install -y libgl1-mesa-glx libglib2.0-0
COPY . .
CMD ["python", "app.py"]
